from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
from stable_baselines3 import PPO

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards import reward_nets
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util import logger as imit_logger
from imitation.util import util

from imitation.scripts.NTRIL.double_integrator.double_integrator import DoubleIntegratorSuboptimalPolicy
from imitation.scripts.SSRR.curve_fit import estimate_airl_returns_by_noise, fit_sigmoid_noise_performance
from imitation.scripts.SSRR.noise_rollouts import generate_noisy_rollout_buckets
from imitation.scripts.SSRR.noisy_airl import EpsilonGreedyActionVecEnvWrapper, NoisyAIRL
from imitation.scripts.SSRR.reward_regression import SSRRRegressor, make_dataloader
from imitation.scripts.SSRR.types import SSRRRegressionConfig


def _make_double_integrator_venv(
    *,
    rng: np.random.Generator,
    n_envs: int,
    env_options: Optional[Mapping[str, Any]] = None,
):
    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"
    return util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=n_envs,
        parallel=False,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        env_make_kwargs=dict(env_options or {}),
    )


def generate_suboptimal_demonstrations(
    *,
    venv,
    rng: np.random.Generator,
    n_episodes: int,
    K_position: float = 0.02,
    K_velocity: float = 0.3,
) -> Sequence[types.TrajectoryWithRew]:
    ghost_env = venv.envs[0]
    subopt = DoubleIntegratorSuboptimalPolicy(
        observation_space=ghost_env.observation_space,
        action_space=ghost_env.action_space,
    )
    subopt.set_K_values(K_position=K_position, K_velocity=K_velocity)
    demos = rollout.rollout(
        subopt,
        venv,
        rollout.make_sample_until(min_episodes=int(n_episodes)),
        rng=rng,
        exclude_infos=True,
    )
    return demos


def run_ssrr_double_integrator(
    *,
    save_dir: str | Path,
    seed: int = 0,
    env_options: Optional[Mapping[str, Any]] = None,
    # Phase 1 (AIRL / Noisy-AIRL)
    phase1_algorithm: str = "airl",  # "airl" | "noisy_airl"
    noisy_airl_epsilon: float = 0.2,
    airl_demo_batch_size: int = 32,
    airl_gen_train_timesteps: int = 256,
    demo_episodes: int = 10,
    airl_total_timesteps: int = 5_000,
    # Phase 1 -> 2
    noise_levels: Sequence[float] = tuple(np.arange(0.0, 1.05, 0.05)),
    rollouts_per_noise: int = 5,
    # Phase 3
    regression_num_samples: int = 5_000,
    regression_steps: int = 3_000,
    regression_batch_size: int = 64,
    regression_cfg: SSRRRegressionConfig = SSRRRegressionConfig(),
    # Final RL on learned reward
    rl_total_timesteps: int = 10_000,
    device: str = "cpu",
):
    """End-to-end SSRR runner for the double integrator (isolation-first).

    Produces artifacts under `save_dir/`:
    - `airl_reward_test.pt` : initial AIRL reward (unshaped base net)
    - `sigmoid_params.npy`  : fitted sigmoid [x0, y0, c, k]
    - `ssrr_reward.pt`      : regressed reward net (torch module)
    - `final_policy.zip`    : PPO policy trained on SSRR reward
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    custom_logger = imit_logger.configure(
        folder=os.path.join(str(save_dir), "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    venv = _make_double_integrator_venv(rng=rng, n_envs=4, env_options=env_options)
    try:
        # -----------------------
        # Phase 1: AIRL bootstrap
        # -----------------------
        demos = generate_suboptimal_demonstrations(venv=venv, rng=rng, n_episodes=demo_episodes)

        airl_reward_net = reward_nets.BasicShapedRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        if phase1_algorithm not in ("airl", "noisy_airl"):
            raise ValueError("phase1_algorithm must be one of: 'airl', 'noisy_airl'")

        # For Noisy-AIRL we sample generator rollouts from pi_eta by injecting
        # epsilon-greedy action noise at the VecEnv boundary during adversarial training.
        venv_airl = venv
        if phase1_algorithm == "noisy_airl":
            venv_airl = EpsilonGreedyActionVecEnvWrapper(
                venv,
                epsilon=float(noisy_airl_epsilon),
                rng=rng,
            )

        gen_algo = PPO("MlpPolicy", venv_airl, verbose=0, device=device)
        trainer_cls = AIRL if phase1_algorithm == "airl" else NoisyAIRL
        trainer_kwargs: Dict[str, Any] = {}
        if phase1_algorithm == "noisy_airl":
            trainer_kwargs["noise_level"] = float(noisy_airl_epsilon)
        trainer = trainer_cls(
            venv=venv_airl,
            demonstrations=demos,
            gen_algo=gen_algo,
            reward_net=airl_reward_net,
            demo_batch_size=int(airl_demo_batch_size),
            gen_train_timesteps=int(airl_gen_train_timesteps),
            log_dir=save_dir,
            custom_logger=custom_logger,
            **trainer_kwargs,
        )
        trainer.train(total_timesteps=int(airl_total_timesteps))

        init_policy = trainer.policy
        init_reward = trainer.reward_test
        th.save(init_reward, save_dir / "airl_reward_test.pt")

        # -----------------------
        # Phase 1.5: noisy rollouts
        # -----------------------
        buckets = generate_noisy_rollout_buckets(
            base_policy=init_policy,
            venv=venv,
            noise_levels=noise_levels,
            n_rollouts_per_noise=rollouts_per_noise,
            rng=rng,
        )

        # -----------------------
        # Phase 2: fit sigma(eta)
        # -----------------------
        perf_data = estimate_airl_returns_by_noise(buckets, init_reward)
        sigmoid_params, diag = fit_sigmoid_noise_performance(perf_data, normalize_y=True, prefer_scipy=True)
        np.save(save_dir / "sigmoid_params.npy", np.asarray(sigmoid_params.as_tuple(), dtype=np.float64))
        np.save(save_dir / "sigmoid_diag_y_true.npy", diag.y_true)
        np.save(save_dir / "sigmoid_diag_y_pred.npy", diag.y_pred)

        # -----------------------
        # Phase 3: reward regression
        # -----------------------
        ssrr_reward = reward_nets.BasicRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            use_state=True,
            use_action=True,
            use_next_state=False,
            use_done=False,
            hid_sizes=(256, 256),
        )
        dl = make_dataloader(
            buckets,
            sigmoid_params,
            num_samples=int(regression_num_samples),
            cfg=regression_cfg,
            batch_size=int(regression_batch_size),
            rng=rng,
        )
        reg = SSRRRegressor(ssrr_reward, lr=1e-4, weight_decay=0.0, device=device)
        reg.train(dl, n_steps=int(regression_steps))
        th.save(ssrr_reward, save_dir / "ssrr_reward.pt")

        # -----------------------
        # Final RL on SSRR reward
        # -----------------------
        reward_venv = RewardVecEnvWrapper(venv, reward_fn=ssrr_reward.predict_processed)
        rl_algo = PPO("MlpPolicy", reward_venv, verbose=0, device=device)
        rl_algo.learn(total_timesteps=int(rl_total_timesteps))
        rl_algo.save(str(save_dir / "final_policy.zip"))

        return {
            "sigmoid_r2": float(diag.r2),
            "sigmoid_params": sigmoid_params.as_tuple(),
        }
    finally:
        venv.close()


if __name__ == "__main__":
    out = run_ssrr_double_integrator(save_dir=Path(__file__).parent / "ssrr_outputs")
    print(out)

