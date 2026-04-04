"""Experimental NTRIL variant for the Double Integrator.

This module keeps baseline NTRIL intact by introducing a subclassed trainer and
an isolated runner with separate output paths.
"""

import argparse
import dataclasses
from datetime import datetime
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from imitation.data import serialize
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import TrajectoryRewardNet
from imitation.scripts.NTRIL.double_integrator.double_integrator import (
    DoubleIntegratorSuboptimalPolicy,
    generate_reference_trajectory,
)
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from imitation.util import logger as imit_logger
from imitation.util import util


@dataclasses.dataclass
class VariantNTRILTrainer(NTRILTrainer):
    """Configurable NTRIL variant.

    This variant supports mixed experimentation points while keeping the
    original `NTRILTrainer` behavior isolated and unchanged.
    """

    ranked_data_source: str = "hybrid"
    include_mpc_step: bool = True
    reward_hidden_sizes: Tuple[int, ...] = (256, 256)

    def train(
        self,
        total_timesteps: int,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        irl_train_kwargs: Optional[Mapping[str, Any]] = None,
        rl_train_kwargs: Optional[Mapping[str, Any]] = None,
        retrain: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, Any]:
        """Run the variant pipeline with optional MPC-step skipping."""
        if retrain == "all":
            force = set(self.STEPS)
        elif retrain is None:
            force: set = set()
        else:
            unknown = set(retrain) - set(self.STEPS)
            if unknown:
                raise ValueError(
                    f"Unknown step name(s) in retrain: {unknown}. "
                    f"Valid names are: {self.STEPS}"
                )
            force = set(retrain)

        stats: Dict[str, Any] = {}

        self._logger.log("Variant Step 1: Obtaining suboptimal policy and initial rollouts...")
        stats["bc"] = self._train_bc_policy(
            force_retrain="bc" in force, **(bc_train_kwargs or {})
        )

        self._logger.log("Variant Step 2: Creating reference trajectory with Tube-Robust MPC...")
        _, mpc_stats = self._extract_reference_trajectory_with_mpc(
            force_retrain="mpc" in force
        )
        stats["mpc"] = {"n_noise_levels": len(mpc_stats)}

        self._logger.log("Variant Step 3: Generating noisy rollouts from RTMPC trajectories...")
        _, rollout_stats = self._generate_noisy_rollouts(
            force_retrain="rollouts" in force
        )
        stats["rollouts"] = rollout_stats

        self._logger.log(
            f"Variant Step 4: Building {self.n_ensemble} ranked datasets "
            f"from '{self.ranked_data_source}' source..."
        )
        self._build_ranked_dataset(force_retrain="ranking" in force)
        stats["ranking"] = {
            f"dataset_{i}_num_samples": len(ds)
            for i, ds in enumerate(self.ranked_datasets)
        }

        self._logger.log(
            f"Variant Step 5: Training {self.n_ensemble} reward networks via IRL..."
        )
        self._train_reward_network(
            force_retrain="irl" in force, **(irl_train_kwargs or {})
        )
        stats["irl"] = {"n_ensemble": self.n_ensemble}

        self._logger.log("Variant Step 6: Training final policy via RL...")
        self._train_final_policy(
            total_timesteps=total_timesteps,
            force_retrain="rl" in force,
            **(rl_train_kwargs or {}),
        )
        stats["rl"] = {"total_timesteps": total_timesteps}

        return stats

    def _extract_reference_trajectory_with_mpc(self, force_retrain: bool = False):
        """Extract nominal reference trajectories with Tube-Robust MPC."""
        rtmpc_dir = os.path.join(self.save_dir, "rtmpc_trajectories")

        # Respect the cache unless forced
        data_exists = (
            os.path.exists(rtmpc_dir)
            and not force_retrain
        )
        if data_exists:
            print("Loading existing nominal reference rtmpc trajectories...")

            rtmpc_path = os.path.join(
                    rtmpc_dir, "reference_trajectory.pkl"
                )
                

            # Load rtmpc trajectories
            rtmpc_trajectories = serialize.load(rtmpc_path)
            self.rtmpc_trajectories.append(rtmpc_trajectories)

            return self.rtmpc_trajectories

        bc_rollouts_path = os.path.join(self.save_dir, "bc_rollouts.pkl")
        # Load bc rollouts, if they don't exist, run the bc training step
        if not os.path.exists(bc_rollouts_path):
            self._train_bc_policy(force_retrain=True)
        with open(bc_rollouts_path, "rb") as f:
            bc_rollouts = pickle.load(f)

        rtmpc_trajectories = []

        for traj in bc_rollouts:
            # Extract physical state from trajectory
            traj_parsed = self._parse_trajectory(traj)
            # Extract original reference trajectory
            original_reference_states = self._extract_reference_states(traj)
            # Apply robust tube MPC to each trajectory to get nominal trajectory
            self.robust_mpc.set_reference_trajectory(traj_parsed)
            rtmpc_trajectory_phys = self._solve_rtmpc(traj_parsed.obs[0], traj_parsed)
            # re append original reference trajectory to the rtmpc_trajectory
            rtmpc_trajectory = self._append_states(rtmpc_trajectory_phys, original_reference_states)
            rtmpc_trajectories.append(rtmpc_trajectory)
        
        serialize.save(os.path.join(rtmpc_dir, "reference_trajectory.pkl"), rtmpc_trajectories)
        self.rtmpc_trajectories = rtmpc_trajectories

        return self.rtmpc_trajectories

    def _generate_noisy_rollouts_from_rtmpc(self, force_retrain: bool = False):
        """Generate noisy rollouts from RTMPC trajectories."""
        noisy_rollouts_path = os.path.join(self.save_dir, "noisy_rollouts.pkl")
        data_exists = (
            os.path.exists(noisy_rollouts_path)
            and not force_retrain
        )
        if data_exists:
            print("Loading existing noisy rollouts from RTMPC trajectories...")
            with open(noisy_rollouts_path, "rb") as f:
                self.noisy_rollouts = pickle.load(f)
            return self.noisy_rollouts
    
        self.noisy_rollouts = []
        for rtmpc_trajectory in self.rtmpc_trajectories:
            for noise_level in self.noise_levels:
                noisy_rollout = self.robust_mpc.constrained_trajectory_augmentation(rtmpc_trajectory, noise_level)
                self.noisy_rollouts[noise_level].append(noisy_rollout)
        serialize.save(noisy_rollouts_path, self.noisy_rollouts)
        return self.noisy_rollouts

    def _get_ranked_dataset_source(self) -> list:
        """Select ranked dataset source: noisy, augmented, or hybrid."""
        source = self.ranked_data_source.lower()
        valid_sources = {"noisy", "augmented", "hybrid"}
        if source not in valid_sources:
            raise ValueError(
                f"Invalid ranked_data_source='{self.ranked_data_source}'. "
                f"Choose one of {sorted(valid_sources)}."
            )

        if source == "noisy":
            return self._ensure_noisy_rollouts_loaded()
        if source == "augmented":
            return self._ensure_augmented_data_loaded()

        noisy_rollouts = self._ensure_noisy_rollouts_loaded()
        augmented_data = self._ensure_augmented_data_loaded()
        if len(noisy_rollouts) != len(augmented_data):
            raise ValueError(
                "Mismatched number of noise buckets between noisy and augmented "
                "data. Regenerate rollouts/augmentation with matching settings."
            )

        hybrid_data = []
        for noisy_bucket, augmented_bucket in zip(noisy_rollouts, augmented_data):
            hybrid_data.append(list(augmented_bucket) + list(noisy_bucket))
        return hybrid_data

    def _make_reward_net(self) -> TrajectoryRewardNet:
        """Construct a reward net with variant-configurable hidden sizes."""
        return TrajectoryRewardNet(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            hid_sizes=self.reward_hidden_sizes,
        )

    def _ensure_noisy_rollouts_loaded(self):
        if self.noisy_rollouts:
            return self.noisy_rollouts

        noisy_rollouts_path = os.path.join(self.save_dir, "noisy_rollouts.pkl")
        if not os.path.exists(noisy_rollouts_path):
            raise ValueError(
                f"Noisy rollouts not found at {noisy_rollouts_path}. "
                "Run _generate_noisy_rollouts() first."
            )
        with open(noisy_rollouts_path, "rb") as f:
            self.noisy_rollouts = pickle.load(f)
        return self.noisy_rollouts

    def _ensure_augmented_data_loaded(self):
        if self.augmented_data:
            return self.augmented_data

        loaded_augmented_data = []
        for noise_level in self.noise_levels:
            path = os.path.join(
                self.save_dir, "augmented_data", f"noise_{noise_level:.2f}.pkl"
            )
            if not os.path.exists(path):
                raise ValueError(
                    f"Augmented data not found at {path}. "
                    "Run _augment_data_with_mpc() first or set "
                    "ranked_data_source='noisy'."
                )
            loaded_augmented_data.append(serialize.load(path))
        self.augmented_data = loaded_augmented_data
        return self.augmented_data


def run_variant_ntril_training(
    env_id: str = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0",
    env_options: Optional[Dict[str, Any]] = None,
    save_dir: str = "./variant_outputs",
    noise_levels: Sequence[float] = tuple(np.arange(0.0, 1.05, 0.05)),
    n_rollouts_per_noise: int = 5,
    n_ensemble: int = 3,
    rl_total_timesteps: int = 1_000_000,
    retrain: Optional[Union[str, Sequence[str]]] = None,
    variant_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[VariantNTRILTrainer, str]:
    """Set up and run the experimental variant pipeline."""
    variant_kwargs = dict(variant_kwargs or {})
    env_options = dict(env_options or {"max_episode_seconds": 200.0, "dt": 1.0})

    print("\n" + "=" * 70)
    print("NTRIL VARIANT PIPELINE — DOUBLEINTEGRATOR-V0")
    print("=" * 70)
    print(f"\nEnvironment : {env_id}")
    print(f"Noise levels: {tuple(noise_levels)}")
    print(f"Save dir    : {save_dir}")
    print(f"Variant cfg : {variant_kwargs}")

    rng = np.random.default_rng(42)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    max_episode_seconds = env_options["max_episode_seconds"]
    dt = env_options["dt"]
    ghost_env = gym.make(env_id, max_episode_seconds=max_episode_seconds, dt=dt)

    ref_mode = "sinusoidal"
    ref_amplitude = 1.0
    ref_frequency = 0.01
    ref_phase = 0.0
    reference_trajectory = generate_reference_trajectory(
        T=ghost_env.max_episode_steps,
        dt=ghost_env.dt,
        mode=ref_mode,
        amplitude=ref_amplitude,
        frequency=ref_frequency,
        phase=ref_phase,
    )
    reference_trajectory_mpc = Trajectory(
        obs=reference_trajectory,
        acts=np.zeros((ghost_env.max_episode_steps, 1)),
        infos=np.array([{}] * ghost_env.max_episode_steps),
        terminal=True,
    )
    np.save(os.path.join(save_dir, "reference_trajectory.npy"), reference_trajectory)
    env_options["reference_trajectory"] = reference_trajectory

    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=5,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        env_make_kwargs=env_options,
    )
    custom_logger = imit_logger.configure(
        folder=os.path.join(save_dir, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=ghost_env.observation_space,
        action_space=ghost_env.action_space,
    )
    suboptimal_policy.set_K_values(K_position=0.02, K_velocity=0.3)

    robust_mpc = RobustTubeMPC(
        horizon=10,
        time_step=dt,
        disturbance_bound=0.1,
        A=ghost_env.A_d,
        B=ghost_env.B_d,
        Q=np.diag([10.0, 1.0]),
        R=0.1 * np.eye(1),
        disturbance_vertices=np.array(
            [[0.1, 0.1], [-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1]]
        ),
        state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds=(np.array([-20.0]), np.array([20.0])),
        reference_trajectory=reference_trajectory_mpc,
        use_approx=True,
    )
    robust_mpc.setup()

    trainer = VariantNTRILTrainer.from_policy(
        policy=suboptimal_policy,
        venv=venv,
        custom_logger=custom_logger,
        noise_levels=tuple(noise_levels),
        n_rollouts_per_noise=n_rollouts_per_noise,
        n_ensemble=n_ensemble,
        irl_batch_size=32,
        irl_lr=1e-3,
        save_dir=save_dir,
        **variant_kwargs,
    )
    trainer.robust_mpc = robust_mpc

    print("\nStarting NTRIL variant training pipeline...")
    stats = trainer.train(
        total_timesteps=rl_total_timesteps,
        retrain=retrain,
    )

    print("\n" + "=" * 70)
    print("NTRIL Variant Training Complete!")
    print("=" * 70)
    print("\nTraining Statistics:")
    for stage, stage_stats in stats.items():
        print(f"\n  {stage.upper()}:")
        if isinstance(stage_stats, dict):
            for key, value in stage_stats.items():
                print(f"    {key}: {value}")
        else:
            print(f"    {stage_stats}")

    venv.close()
    ghost_env.close()
    return trainer, f"{ref_mode}_A{ref_amplitude}_f{ref_frequency}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated experimental NTRIL variant pipeline.",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=None)
    parser.add_argument("--n-rollouts-per-noise", type=int, default=5)
    parser.add_argument("--rl-total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--retrain",
        nargs="*",
        default=None,
        help="Set to 'all' or provide steps like: rollouts mpc ranking irl rl",
    )
    parser.add_argument(
        "--ranked-data-source",
        type=str,
        choices=("hybrid", "augmented", "noisy"),
        default="hybrid",
    )
    parser.add_argument(
        "--disable-mpc-step",
        action="store_true",
        help="Skip MPC augmentation and rely on ranked_data_source.",
    )
    parser.add_argument(
        "--reward-hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256, 128],
    )
    parser.add_argument("--archive-name", type=str, default=None)
    return parser


def main():
    args = _build_parser().parse_args()

    script_dir = Path(__file__).parent.resolve()
    save_dir = (
        Path(args.save_dir).resolve()
        if args.save_dir is not None
        else script_dir / "variant_outputs"
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

    variant_kwargs = {
        "ranked_data_source": args.ranked_data_source,
        "include_mpc_step": not args.disable_mpc_step,
        "reward_hidden_sizes": tuple(args.reward_hidden_sizes),
    }

    trainer, ref_tag = run_variant_ntril_training(
        save_dir=str(save_dir),
        noise_levels=noise_levels,
        n_rollouts_per_noise=args.n_rollouts_per_noise,
        rl_total_timesteps=args.rl_total_timesteps,
        retrain=retrain,
        variant_kwargs=variant_kwargs,
    )

    archive_name = (
        args.archive_name
        if args.archive_name
        else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ref_tag}_variant"
    )
    trainer.archive_run(
        name=archive_name,
        archive_root=str(save_dir / "archived_runs"),
    )
    print(f"\nAll variant outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
