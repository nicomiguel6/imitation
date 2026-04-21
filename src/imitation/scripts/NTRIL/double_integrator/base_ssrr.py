"""SSRR baseline for the Double Integrator System.

Implements SSRR (Self-Supervised Reward Regression) as an alternative to D-REX
for the "learning from suboptimal demonstrations" portion.

This is intentionally implemented as a **subclass** of `NTRILTrainer` to keep
integration minimal and localized (mirroring `DREXTrainer`).

Pipeline (high level):
  Step 1: obtain suboptimal policy (BC or provided policy)
  Step 1b: train AIRL/Noisy-AIRL on suboptimal rollouts to bootstrap (pi_tilde, R_tilde)
  Step 2: generate noisy rollouts from pi_tilde (epsilon-greedy action noise)
  Step 3: fit sigmoid sigma(eta) to AIRL-evaluated noisy rollouts
  Step 4: regress reward R_theta on snippets (SSRR loss)
  Step 5: train final policy via RL using ensemble-averaged SSRR reward
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch as th
from stable_baselines3 import PPO

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards import reward_nets
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.double_integrator.double_integrator import DoubleIntegratorSuboptimalPolicy
from imitation.util import logger as imit_logger
from imitation.util import util

from imitation.scripts.SSRR.curve_fit import estimate_airl_returns_by_noise, fit_sigmoid_noise_performance
from imitation.scripts.SSRR.noise_rollouts import generate_noisy_rollout_buckets
from imitation.scripts.SSRR.noisy_airl import EpsilonGreedyActionVecEnvWrapper, NoisyAIRL
from imitation.scripts.SSRR.reward_regression import SSRRRegressor, make_dataloader
from imitation.scripts.SSRR.types import SSRRRegressionConfig


@dataclasses.dataclass
class SSRRTrainer(NTRILTrainer):
    """SSRR trainer integrated into the NTRIL codebase."""

    # SSRR config
    phase1_algorithm: str = "noisy_airl"  # "airl" | "noisy_airl"
    noisy_airl_epsilon: float = 0.2
    airl_total_timesteps: int = 50_000
    airl_demo_batch_size: int = 32
    airl_gen_train_timesteps: int = 2048
    airl_demo_rollouts: int = 10

    regression_num_samples: int = 5_000
    regression_steps: int = 3_000
    regression_batch_size: int = 64
    regression_cfg: SSRRRegressionConfig = dataclasses.field(default_factory=SSRRRegressionConfig)

    def train(
        self,
        total_timesteps: int,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        irl_train_kwargs: Optional[Mapping[str, Any]] = None,
        rl_train_kwargs: Optional[Mapping[str, Any]] = None,
        retrain: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, Any]:
        del irl_train_kwargs  # SSRR replaces ranked-IRL reward learning

        if retrain == "all":
            force = {"bc", "airl", "rollouts", "ssrr", "rl"}
        elif retrain is None:
            force = set()
        else:
            force = set(retrain)

        stats: Dict[str, Any] = {}

        self._logger.log("Step 1: Obtaining suboptimal policy (BC / provided)...")
        stats["bc"] = self._train_bc_policy(force_retrain="bc" in force, **(bc_train_kwargs or {}))

        self._logger.log("Step 1b: Training AIRL/Noisy-AIRL bootstrap...")
        stats["airl"] = self._train_airl_bootstrap(force_retrain="airl" in force)

        self._logger.log("Step 2: Generating noisy rollouts from AIRL policy...")
        buckets = self._generate_noisy_rollouts_from_airl(force_retrain="rollouts" in force)
        stats["rollouts"] = {"n_buckets": len(buckets), "noise_levels": [b.noise_level for b in buckets]}

        self._logger.log("Step 3+4: Fitting sigmoid + SSRR reward regression...")
        stats["ssrr"] = self._train_ssrr_reward(buckets, force_retrain="ssrr" in force)

        self._logger.log("Step 5: Training final policy via RL on SSRR reward...")
        stats["rl"] = self._train_final_policy(
            total_timesteps=total_timesteps,
            force_retrain="rl" in force,
            **(rl_train_kwargs or {}),
        )
        return stats

    def _train_airl_bootstrap(self, force_retrain: bool = False) -> Dict[str, Any]:
        save_path = os.path.join(self.save_dir, "ssrr_airl_reward_test.pt")
        if os.path.exists(save_path) and not force_retrain:
            self._ssrr_airl_reward = th.load(save_path, map_location="cpu")
            # policy is not persisted here (kept simple); we will retrain if missing
            return {"loaded_reward": True}

        if self.phase1_algorithm not in ("airl", "noisy_airl"):
            raise ValueError("phase1_algorithm must be one of: 'airl', 'noisy_airl'")

        # Collect suboptimal rollouts to use as AIRL demonstrations.
        demos = rollout.rollout(
            self.policy,
            self.venv,
            rollout.make_sample_until(min_episodes=int(self.airl_demo_rollouts)),
            rng=self.rng,
            exclude_infos=True,
            reset_options={"reference_trajectory": self.reference_trajectory}
            if self.reference_trajectory is not None
            else None,
        )

        airl_reward_net = reward_nets.BasicShapedRewardNet(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
        )

        venv_airl = self.venv
        if self.phase1_algorithm == "noisy_airl":
            venv_airl = EpsilonGreedyActionVecEnvWrapper(
                self.venv,
                epsilon=float(self.noisy_airl_epsilon),
                rng=self.rng,
            )

        gen_algo = PPO("MlpPolicy", venv_airl, verbose=0, device=self.device)
        trainer_cls = AIRL if self.phase1_algorithm == "airl" else NoisyAIRL
        trainer_kwargs: Dict[str, Any] = {}
        if self.phase1_algorithm == "noisy_airl":
            trainer_kwargs["noise_level"] = float(self.noisy_airl_epsilon)

        trainer = trainer_cls(
            venv=venv_airl,
            demonstrations=demos,
            gen_algo=gen_algo,
            reward_net=airl_reward_net,
            demo_batch_size=int(self.airl_demo_batch_size),
            gen_train_timesteps=int(self.airl_gen_train_timesteps),
            log_dir=Path(self.save_dir),
            custom_logger=self.custom_logger,
            **trainer_kwargs,
        )
        trainer.train(total_timesteps=int(self.airl_total_timesteps))

        self._ssrr_airl_policy = trainer.policy
        self._ssrr_airl_reward = trainer.reward_test
        th.save(self._ssrr_airl_reward, save_path)
        return {"loaded_reward": False}

    def _generate_noisy_rollouts_from_airl(self, force_retrain: bool = False):
        save_path = os.path.join(self.save_dir, "ssrr_noisy_buckets.pkl")
        if os.path.exists(save_path) and not force_retrain:
            from imitation.data import serialize

            return serialize.load(save_path)

        if not hasattr(self, "_ssrr_airl_policy"):
            raise ValueError("AIRL policy not available; run _train_airl_bootstrap first.")

        buckets = generate_noisy_rollout_buckets(
            base_policy=self._ssrr_airl_policy,
            venv=self.venv,
            noise_levels=self.noise_levels,
            n_rollouts_per_noise=self.n_rollouts_per_noise,
            rng=self.rng,
        )
        from imitation.data import serialize

        serialize.save(save_path, buckets)
        return buckets

    def _train_ssrr_reward(self, buckets, force_retrain: bool = False) -> Dict[str, Any]:
        # Cache ensemble SSRR rewards in ensemble/ just like NTRIL.
        ensemble_dir = os.path.join(self.save_dir, "ensemble_ssrr")
        os.makedirs(ensemble_dir, exist_ok=True)

        if not hasattr(self, "_ssrr_airl_reward"):
            raise ValueError("AIRL reward not available; run _train_airl_bootstrap first.")

        perf_data = estimate_airl_returns_by_noise(buckets, self._ssrr_airl_reward)
        sigmoid_params, diag = fit_sigmoid_noise_performance(perf_data, normalize_y=True, prefer_scipy=True)
        np.save(os.path.join(ensemble_dir, "sigmoid_params.npy"), np.asarray(sigmoid_params.as_tuple(), dtype=np.float64))

        # Train SSRR reward ensemble
        self.reward_nets_ensemble = []
        for i in range(self.n_ensemble):
            ckpt = os.path.join(ensemble_dir, f"ssrr_reward_{i}.pt")
            if os.path.exists(ckpt) and not force_retrain:
                net = th.load(ckpt, map_location=self.device)
                self.reward_nets_ensemble.append(net)
                continue

            net = reward_nets.BasicRewardNet(
                observation_space=self.venv.observation_space,
                action_space=self.venv.action_space,
                use_state=True,
                use_action=True,
                use_next_state=False,
                use_done=False,
                hid_sizes=(256, 256),
            )
            dl = make_dataloader(
                buckets,
                sigmoid_params,
                num_samples=int(self.regression_num_samples),
                cfg=self.regression_cfg,
                batch_size=int(self.regression_batch_size),
                rng=np.random.default_rng(i),
            )
            reg = SSRRRegressor(net, lr=1e-4, weight_decay=0.0, device=self.device)
            reg.train(dl, n_steps=int(self.regression_steps))
            th.save(net, ckpt)
            self.reward_nets_ensemble.append(net)

        # NTRIL’s _train_final_policy uses reward_nets_ensemble; we’re done.
        return {"sigmoid_r2": float(diag.r2), "n_ensemble": self.n_ensemble}


def run_ssrr_training(
    *,
    env_id: str = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0",
    env_options: Optional[Dict[str, Any]] = None,
    save_dir: str = "./ssrr_outputs",
    noise_levels: tuple = tuple(np.arange(0.0, 1.05, 0.05)),
    n_rollouts_per_noise: int = 5,
    n_ensemble: int = 3,
    rl_total_timesteps: int = 1_000_000,
    retrain: Optional[Union[str, Sequence[str]]] = None,
) -> SSRRTrainer:
    rng = np.random.default_rng(42)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=5,
        parallel=False,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        env_make_kwargs=dict(env_options or {}),
    )
    custom_logger = imit_logger.configure(
        folder=os.path.join(save_dir, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    ghost_env = venv.envs[0]
    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=ghost_env.observation_space,
        action_space=ghost_env.action_space,
    )
    suboptimal_policy.set_K_values(K_position=0.02, K_velocity=0.3)

    trainer = SSRRTrainer.from_policy(
        policy=suboptimal_policy,
        venv=venv,
        custom_logger=custom_logger,
        noise_levels=noise_levels,
        n_rollouts_per_noise=n_rollouts_per_noise,
        n_ensemble=n_ensemble,
        save_dir=save_dir,
    )

    trainer.train(total_timesteps=rl_total_timesteps, retrain=retrain)
    venv.close()
    return trainer

