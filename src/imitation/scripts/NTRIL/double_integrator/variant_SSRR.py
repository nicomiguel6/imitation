"""SSRR variant pipeline integrated with NTRIL-style RTMPC augmentation.

Pipeline (requested):
  1. Train AIRL / Noisy-AIRL from suboptimal data to get coarse reward + policy.
  2. Inject action noise into that policy and collect noise-conditioned rollouts.
  3. Augment each noise bucket with RTMPC trajectory snippets.
  4. Label trajectories with coarse reward, fit noise-performance sigmoid, then
     regress the final reward network(s) with SSRR.
  5. Train final RL policy on the learned reward.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common import policies

from imitation.algorithms import bc
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout, serialize, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards import reward_nets
from imitation.scripts.NTRIL.double_integrator.double_integrator import (
    DoubleIntegratorSuboptimalPolicy,
    generate_reference_trajectory,
)
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from imitation.scripts.SSRR.curve_fit import (
    estimate_airl_returns_by_noise,
    fit_sigmoid_noise_performance,
)
from imitation.scripts.SSRR.noise_rollouts import generate_noisy_rollout_buckets
from imitation.scripts.SSRR.noisy_airl import EpsilonGreedyActionVecEnvWrapper, NoisyAIRL
from imitation.scripts.SSRR.reward_regression import SSRRRegressor, make_dataloader
from imitation.scripts.SSRR.types import NoiseBucket, SSRRRegressionConfig
from imitation.util import logger as imit_logger
from imitation.util import util


@dataclasses.dataclass
class VariantSSRRTrainer(NTRILTrainer):
    """SSRR variant that reuses NTRIL RTMPC augmentation in Step 3."""

    phase1_algorithm: str = "noisy_airl"  # "airl" | "noisy_airl"
    noisy_airl_epsilon: float = 0.2
    airl_total_timesteps: int = 30_000
    airl_demo_batch_size: int = 32
    airl_gen_train_timesteps: int = 2048
    airl_demo_rollouts: int = 10

    regression_num_samples: int = 5_000
    regression_steps: int = 3_000
    regression_batch_size: int = 64
    regression_cfg: SSRRRegressionConfig = dataclasses.field(
        default_factory=SSRRRegressionConfig,
    )

    STEPS: tuple = ("airl", "rollouts", "mpc", "ssrr", "rl")

    def train(
        self,
        total_timesteps: int,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        irl_train_kwargs: Optional[Mapping[str, Any]] = None,
        rl_train_kwargs: Optional[Mapping[str, Any]] = None,
        retrain: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, Any]:
        del bc_train_kwargs
        del irl_train_kwargs

        if retrain == "all":
            force = set(self.STEPS)
        elif retrain is None:
            force = set()
        else:
            unknown = set(retrain) - set(self.STEPS)
            if unknown:
                raise ValueError(
                    f"Unknown step name(s) in retrain: {unknown}. "
                    f"Valid names are: {self.STEPS}",
                )
            force = set(retrain)

        stats: Dict[str, Any] = {}

        self._logger.log("Variant SSRR Step 1: Training AIRL/Noisy-AIRL bootstrap...")
        stats["airl"] = self._train_airl_bootstrap(force_retrain="airl" in force)

        self._logger.log("Variant SSRR Step 2: Generating noisy rollouts from AIRL policy...")
        buckets = self._generate_noisy_rollout_buckets(force_retrain="rollouts" in force)
        stats["rollouts"] = {
            "n_noise_levels": len(buckets),
            "rollouts_per_noise": [len(b.trajectories) for b in buckets],
        }

        self._logger.log("Variant SSRR Step 3: Augmenting rollouts via RTMPC snippets...")
        if not hasattr(self, "_robust_mpc"):
            raise ValueError(
                "robust_mpc must be set before Step 3. "
                "Pass robust_mpc=... into run_variant_ssrr_training().",
            )
        self._augment_data_with_mpc(force_retrain="mpc" in force)
        stats["mpc"] = {
            "augmented_per_noise": [len(x) for x in self.augmented_data],
        }

        self._logger.log(
            "Variant SSRR Step 4: Coarse-reward labeling, sigmoid fit, and SSRR regression...",
        )
        stats["ssrr"] = self._train_ssrr_reward_from_combined_data(
            force_retrain="ssrr" in force,
        )

        self._logger.log("Variant SSRR Step 5: Training final RL policy...")
        self._train_final_policy(
            total_timesteps=total_timesteps,
            force_retrain="rl" in force,
            **(rl_train_kwargs or {}),
        )
        stats["rl"] = {"total_timesteps": int(total_timesteps)}

        return stats

    def _train_airl_bootstrap(self, force_retrain: bool = False) -> Dict[str, Any]:
        reward_path = os.path.join(self.save_dir, "ssrr_variant_airl_reward.pt")
        policy_dir = os.path.join(self.save_dir, "ssrr_variant_airl_policy")

        if (
            os.path.exists(reward_path)
            and os.path.exists(policy_dir)
            and not force_retrain
        ):
            self._coarse_reward_net = th.load(reward_path, map_location=self.device)
            self._airl_policy = bc.reconstruct_policy(policy_dir, device=self.device)
            self.bc_policy = self._airl_policy
            return {"loaded_from_cache": True}

        if self.phase1_algorithm not in ("airl", "noisy_airl"):
            raise ValueError("phase1_algorithm must be one of: 'airl', 'noisy_airl'")

        if self._expert_mode.name == "DEMONSTRATIONS":
            demos = self._demonstrations
        else:
            demos = rollout.rollout(
                self._suboptimal_policy,
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

        self._airl_policy = trainer.policy
        self._coarse_reward_net = trainer.reward_test
        self.bc_policy = self._airl_policy

        th.save(self._coarse_reward_net, reward_path)
        util.save_policy(self._airl_policy, policy_dir)
        return {"loaded_from_cache": False}

    def _generate_noisy_rollout_buckets(
        self,
        force_retrain: bool = False,
    ) -> Sequence[NoiseBucket]:
        buckets_path = os.path.join(self.save_dir, "ssrr_variant_noisy_buckets.pkl")
        if os.path.exists(buckets_path) and not force_retrain:
            buckets = serialize.load(buckets_path)
            self._sync_noisy_rollouts_from_buckets(buckets)
            return buckets

        if not hasattr(self, "_airl_policy"):
            raise ValueError("AIRL policy not available. Run _train_airl_bootstrap() first.")

        buckets = generate_noisy_rollout_buckets(
            base_policy=self._airl_policy,
            venv=self.venv,
            noise_levels=self.noise_levels,
            n_rollouts_per_noise=self.n_rollouts_per_noise,
            rng=self.rng,
        )
        serialize.save(buckets_path, buckets)
        self._sync_noisy_rollouts_from_buckets(buckets)
        return buckets

    def _sync_noisy_rollouts_from_buckets(self, buckets: Sequence[NoiseBucket]) -> None:
        self.noisy_rollouts = [list(b.trajectories) for b in buckets]

    def _build_combined_buckets(self) -> Sequence[NoiseBucket]:
        if not self.noisy_rollouts:
            raise ValueError("No noisy rollouts found; run step 2 first.")
        if not self.augmented_data:
            raise ValueError("No augmented data found; run step 3 first.")

        combined: list[NoiseBucket] = []
        for idx, noise_level in enumerate(self.noise_levels):
            original_trajs = list(self.noisy_rollouts[idx])
            augmented_trajs = list(self.augmented_data[idx])
            combined.append(
                NoiseBucket(
                    noise_level=float(noise_level),
                    trajectories=original_trajs + augmented_trajs,
                ),
            )
        return combined

    def _train_ssrr_reward_from_combined_data(
        self,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        if not hasattr(self, "_coarse_reward_net"):
            raise ValueError(
                "Coarse AIRL reward is unavailable; run _train_airl_bootstrap() first.",
            )

        ensemble_dir = os.path.join(self.save_dir, "ensemble_ssrr_variant")
        os.makedirs(ensemble_dir, exist_ok=True)

        buckets = self._build_combined_buckets()
        perf_data = estimate_airl_returns_by_noise(buckets, self._coarse_reward_net)
        sigmoid_params, diag = fit_sigmoid_noise_performance(
            perf_data,
            normalize_y=True,
            prefer_scipy=True,
        )

        np.save(
            os.path.join(ensemble_dir, "sigmoid_params.npy"),
            np.asarray(sigmoid_params.as_tuple(), dtype=np.float64),
        )
        np.save(os.path.join(ensemble_dir, "sigmoid_diag_y_true.npy"), diag.y_true)
        np.save(os.path.join(ensemble_dir, "sigmoid_diag_y_pred.npy"), diag.y_pred)

        self.reward_nets_ensemble = []
        for i in range(self.n_ensemble):
            ckpt = os.path.join(ensemble_dir, f"reward_net_{i}.pth")
            if os.path.exists(ckpt) and not force_retrain:
                net = self._make_ssrr_reward_net()
                net.load_state_dict(th.load(ckpt, map_location=self.device))
                net.to(self.device)
                self.reward_nets_ensemble.append(net)
                continue

            net = self._make_ssrr_reward_net()
            dataloader = make_dataloader(
                buckets,
                sigmoid_params,
                num_samples=int(self.regression_num_samples),
                cfg=self.regression_cfg,
                batch_size=int(self.regression_batch_size),
                rng=np.random.default_rng(i),
            )
            reg = SSRRRegressor(net, lr=1e-4, weight_decay=0.0, device=self.device)
            reg.train(dataloader, n_steps=int(self.regression_steps))
            th.save(net.state_dict(), ckpt)
            self.reward_nets_ensemble.append(net)

        return {
            "sigmoid_r2": float(diag.r2),
            "n_ensemble": int(self.n_ensemble),
            "total_trajectories_per_noise": [len(b.trajectories) for b in buckets],
        }

    def _make_ssrr_reward_net(self) -> reward_nets.BasicRewardNet:
        return reward_nets.BasicRewardNet(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            use_state=True,
            use_action=True,
            use_next_state=False,
            use_done=False,
            hid_sizes=(256, 256),
        )


def run_variant_ssrr_training(
    *,
    suboptimal_policy: Optional[policies.BasePolicy] = None,
    demonstrations: Optional[Sequence[types.TrajectoryWithRew]] = None,
    env_id: str = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0",
    env_options: Optional[Dict[str, Any]] = None,
    save_dir: str = "./variant_ssrr_outputs",
    noise_levels: Sequence[float] = tuple(np.arange(0.0, 1.05, 0.05)),
    n_rollouts_per_noise: int = 5,
    n_ensemble: int = 3,
    rl_total_timesteps: int = 1_000_000,
    retrain: Optional[Union[str, Sequence[str]]] = None,
    variant_kwargs: Optional[Mapping[str, Any]] = None,
    robust_mpc: Optional[RobustTubeMPC] = None,
    reference_trajectory: Optional[np.ndarray] = None,
) -> VariantSSRRTrainer:
    """Run SSRR variant pipeline with optional RTMPC augmentation."""
    if suboptimal_policy is None and demonstrations is None:
        raise ValueError("Either 'suboptimal_policy' or 'demonstrations' must be provided.")

    variant_kwargs = dict(variant_kwargs or {})
    env_options = dict(env_options or {"max_episode_seconds": 200.0, "dt": 1.0})
    env_options["reference_trajectory"] = reference_trajectory

    print("\n" + "=" * 70)
    print("SSRR VARIANT PIPELINE — DOUBLEINTEGRATOR-V0")
    print("=" * 70)
    print(f"\nEnvironment : {env_id}")
    print(f"Noise levels: {tuple(noise_levels)}")
    print(f"Save dir    : {save_dir}")
    print(f"Variant cfg : {variant_kwargs}")

    rng = np.random.default_rng(42)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        env_make_kwargs=env_options,
    )
    custom_logger = imit_logger.configure(
        folder=os.path.join(save_dir, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    common_kwargs = dict(
        venv=venv,
        custom_logger=custom_logger,
        noise_levels=tuple(noise_levels),
        n_rollouts_per_noise=n_rollouts_per_noise,
        n_ensemble=n_ensemble,
        save_dir=save_dir,
        reference_trajectory=reference_trajectory,
        **variant_kwargs,
    )
    if suboptimal_policy is not None:
        trainer = VariantSSRRTrainer.from_policy(suboptimal_policy, **common_kwargs)
    else:
        trainer = VariantSSRRTrainer.from_demonstrations(demonstrations, **common_kwargs)

    if robust_mpc is not None:
        trainer.robust_mpc = robust_mpc

    trainer.train(total_timesteps=rl_total_timesteps, retrain=retrain)
    venv.close()
    return trainer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SSRR+NTRIL RTMPC variant for double integrator.",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=None)
    parser.add_argument("--n-rollouts-per-noise", type=int, default=5)
    parser.add_argument("--n-ensemble", type=int, default=3)
    parser.add_argument("--rl-total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--retrain", nargs="*", default=None)
    parser.add_argument(
        "--phase1-algorithm",
        type=str,
        choices=("airl", "noisy_airl"),
        default="noisy_airl",
    )
    parser.add_argument("--noisy-airl-epsilon", type=float, default=0.2)
    parser.add_argument("--airl-total-timesteps", type=int, default=30_000)
    parser.add_argument("--regression-steps", type=int, default=3_000)
    parser.add_argument("--regression-num-samples", type=int, default=5_000)
    parser.add_argument("--archive-name", type=str, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    script_dir = Path(__file__).parent.resolve()
    save_dir = (
        Path(args.save_dir).resolve()
        if args.save_dir is not None
        else script_dir / "variant_ssrr_outputs"
    )
    noise_levels = (
        tuple(args.noise_levels)
        if args.noise_levels is not None
        else tuple(np.arange(0.0, 1.05, 0.05))
    )
    retrain: Optional[Union[str, Sequence[str]]] = args.retrain
    if retrain == ["all"]:
        retrain = "all"
    elif retrain == []:
        retrain = None

    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"
    env_opts = {"max_episode_seconds": 200.0, "dt": 1.0}

    ghost_env = gym.make(env_id, **env_opts)
    reference_trajectory = generate_reference_trajectory(
        T=ghost_env.max_episode_steps,
        dt=ghost_env.dt,
        mode="sinusoidal",
        amplitude=1.0,
        frequency=0.01,
        phase=0.0,
    )
    reference_trajectory_mpc = types.Trajectory(
        obs=reference_trajectory,
        acts=np.zeros((ghost_env.max_episode_steps, 1)),
        infos=np.array([{}] * ghost_env.max_episode_steps),
        terminal=True,
    )

    robust_tube_mpc = RobustTubeMPC(
        horizon=10,
        time_step=env_opts["dt"],
        A=ghost_env.A_d,
        B=ghost_env.B_d,
        Q=np.diag([10.0, 1.0]),
        R=0.1 * np.eye(1),
        disturbance_vertices=np.array(
            [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1]],
        ),
        state_bounds=(np.array([-1000.0, -1000.0]), np.array([1000.0, 1000.0])),
        control_bounds=(np.array([-5.0]), np.array([5.0])),
        reference_trajectory=reference_trajectory_mpc,
        use_approx=True,
    )
    robust_tube_mpc.setup()

    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=ghost_env.observation_space,
        action_space=ghost_env.action_space,
    )
    suboptimal_policy.set_K_values(0.02, 0.3)

    variant_kwargs = {
        "phase1_algorithm": args.phase1_algorithm,
        "noisy_airl_epsilon": args.noisy_airl_epsilon,
        "airl_total_timesteps": args.airl_total_timesteps,
        "regression_steps": args.regression_steps,
        "regression_num_samples": args.regression_num_samples,
    }
    trainer = run_variant_ssrr_training(
        suboptimal_policy=suboptimal_policy,
        env_id=env_id,
        env_options=env_opts,
        save_dir=str(save_dir),
        noise_levels=noise_levels,
        n_rollouts_per_noise=args.n_rollouts_per_noise,
        n_ensemble=args.n_ensemble,
        rl_total_timesteps=args.rl_total_timesteps,
        retrain=retrain,
        variant_kwargs=variant_kwargs,
        robust_mpc=robust_tube_mpc,
        reference_trajectory=reference_trajectory,
    )
    archive_name = (
        args.archive_name
        if args.archive_name
        else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ssrr_variant"
    )
    trainer.archive_run(
        name=archive_name,
        archive_root=str(save_dir / "archived_runs"),
    )
    ghost_env.close()
    print(f"\nAll SSRR variant outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
