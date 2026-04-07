"""Experimental NTRIL variant for the Double Integrator.

This module keeps baseline NTRIL intact by introducing a subclassed trainer and
an isolated runner with separate output paths.
"""

import argparse
import dataclasses
from tqdm import tqdm
from datetime import datetime
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from imitation.data import serialize
from imitation.data.types import Trajectory, TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import TrajectoryRewardNet
from imitation.scripts.NTRIL.double_integrator.double_integrator import (
    DoubleIntegratorSuboptimalPolicy,
    generate_reference_trajectory,
)
from imitation.policies.base import NonTrainablePolicy
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
        rtmpc_trajectories = self._extract_reference_trajectory_with_mpc(
            force_retrain="mpc" in force
        )
        # stats["mpc"] = {"n_noise_levels": len(mpc_stats)}

        self._logger.log("Variant Step 3: Generating noisy augmented rollouts from RTMPC trajectories...")
        noisy_rollouts = self._generate_noisy_rollouts_from_rtmpc(
            force_retrain="rollouts" in force
        )
        # stats["rollouts"] = noisy_rollouts

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
            self.rtmpc_trajectories = rtmpc_trajectories

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
        noisy_rollouts_path = os.path.join(self.save_dir, "noisy_rollouts")
        data_exists = (
            os.path.exists(noisy_rollouts_path)
            and not force_retrain
        )
        if data_exists:
            print("Loading existing noisy rollouts from RTMPC trajectories...")
            for noise_level in self.noise_levels:
                noisy_rollouts_path_for_noise_level = os.path.join(noisy_rollouts_path, f"noise_{noise_level:.2f}.pkl")
                if os.path.exists(noisy_rollouts_path_for_noise_level):
                    noisy_rollouts_for_noise_level = serialize.load(noisy_rollouts_path_for_noise_level)
                    self.noisy_rollouts.append(noisy_rollouts_for_noise_level)
                else:
                    raise ValueError(f"Noisy rollouts not found at {noisy_rollouts_path_for_noise_level}. Run _generate_noisy_rollouts_from_rtmpc() first.")
            
            return self.noisy_rollouts
    
        # Noisy rollout process
        self.noisy_rollouts = []
        for noise_level in self.noise_levels:
            noisy_rollouts_for_noise_level = []
            for rtmpc_trajectory in tqdm(self.rtmpc_trajectories, desc="Generating noisy rollouts from RTMPC trajectories"):
                # extract reference states from the rtmpc_trajectory
                reference_states = self._extract_reference_states(rtmpc_trajectory)
                # extract physical state from the rtmpc_trajectory
                physical_state = self._parse_trajectory(rtmpc_trajectory)
                noisy_rollouts = self.robust_mpc.constrained_trajectory_augmentation(physical_state, reference_states, noise_level)
                noisy_rollouts_for_noise_level.extend(noisy_rollouts)
            
            # # plot the noisy rollouts
            # for noisy_rollout in noisy_rollouts_for_noise_level:
            #     plt.plot(noisy_rollout.obs[:, 0])
            # # plot reference trajectory
            # plt.plot(self.reference_trajectory[:, 0], 'k--', linewidth=4, label="Reference Trajectory")
            # plt.legend()
            # plt.xlabel("Time")
            # plt.ylabel("State")
            # plt.title(f"Noisy Rollouts for Noise Level {noise_level:.2f}")
            # plt.savefig(os.path.join(self.save_dir, "noisy_rollouts_for_noise_level", f"noise_{noise_level:.2f}.png"))
            # plt.close()
            # Save noisy rollouts for each noise level to a file
            serialize.save(os.path.join(self.save_dir, "noisy_rollouts", f"noise_{noise_level:.2f}.pkl"), noisy_rollouts_for_noise_level)
            self.noisy_rollouts.append(noisy_rollouts_for_noise_level)

        return self.noisy_rollouts
    
    def _get_ranked_dataset_source(self) -> list:
        """Return the trajectory data used to build the ranked dataset.

        Returns:
            List of noisy rollouts for each noise level
        """
        if not self.noisy_rollouts:
            for noise_level in self.noise_levels:
                path = os.path.join(
                    self.save_dir, "noisy_rollouts_for_noise_level", f"noise_{noise_level:.2f}.pkl"
                )
                if os.path.exists(path):
                    self.noisy_rollouts.append(serialize.load(path))
                else:
                    raise ValueError(
                        f"Noisy rollouts not found at {path}. "
                        "Run _generate_noisy_rollouts_from_rtmpc() first."
                    )
        return self.noisy_rollouts

    def _make_reward_net(self) -> TrajectoryRewardNet:
        """Construct a reward net with variant-configurable hidden sizes."""
        return TrajectoryRewardNet(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            hid_sizes=self.reward_hidden_sizes,
        )

    def _solve_rtmpc(self, initial_state: np.ndarray, reference_trajectory: Trajectory) -> TrajectoryWithRew:
        """Run the Robust Tube MPC over a full trajectory using :meth:`RobustTubeMPC.solve_mpc`.

        Args:
            initial_state: Initial observation (may be augmented; the physical
                slice is extracted automatically by ``reset_episode``).
            reference_trajectory: Noisy rollout trajectory that defines the
                episode length and carries noise metadata in its ``infos``.

        Returns:
            RTMPC nominal trajectory as a numpy array
        """
        # Reset MPC state machine for a fresh episode (sets z_nominal, warm-starts NLP).
        self.robust_mpc.reset_episode(initial_state)

        phys_initial = self.robust_mpc._extract_physical_state(initial_state)
        builder = util.TrajectoryBuilder()
        builder.start_episode(initial_obs=phys_initial)

        state = phys_initial
        n_steps = len(reference_trajectory.obs) - 1
        for itr in range(n_steps):
            next_state, applied_u, _ = self.robust_mpc.solve_mpc(state)
            builder.add_step(
                action=applied_u.flatten(),
                next_obs=next_state.flatten(),
                reward=0.0,
                info={} if reference_trajectory.infos is None else reference_trajectory.infos[itr],
            )
            state = next_state

        return builder.finish()


def run_variant_ntril_training(
    suboptimal_policy: Optional[NonTrainablePolicy] = None,
    demonstrations: Optional[Sequence[TrajectoryWithRew]] = None,
    env_id: str = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0",
    env_options: Optional[Dict[str, Any]] = None,
    save_dir: str = "./variant_outputs",
    noise_levels: Sequence[float] = tuple(np.arange(0.0, 1.05, 0.05)),
    n_rollouts_per_noise: int = 5,
    n_ensemble: int = 3,
    rl_total_timesteps: int = 1_000_000,
    run_individual_steps: Optional[list] = None,
    retrain: Optional[Union[str, Sequence[str]]] = None,
    variant_kwargs: Optional[Mapping[str, Any]] = None,
    robust_mpc: Optional[RobustTubeMPC] = None,
    reference_trajectory: Optional[np.ndarray] = None,
) -> "VariantNTRILTrainer":
    """Set up and run the experimental variant pipeline.

    Mirrors ``run_ntril_training`` in structure, running individual numbered
    steps (1–6) in order and respecting the ``retrain`` cache-bypass flags.

    Pipeline steps:
      1: Obtain suboptimal policy (BC training or skip if policy provided)
      2: Extract nominal reference trajectory via Robust Tube MPC
      3: Generate noisy rollouts from RTMPC trajectories
      4: Build ranked datasets (ensemble)
      5: Train reward networks via demonstration-ranked IRL
      6: Train final policy via RL on ensemble reward

    Args:
        suboptimal_policy: Pre-trained policy (mutually exclusive with
            ``demonstrations``); skips Step 1 if provided.
        demonstrations: Trajectory data for BC training in Step 1 (mutually
            exclusive with ``suboptimal_policy``).
        env_id: Gymnasium environment ID.
        env_options: Dict of keyword arguments forwarded to ``gym.make``.
        save_dir: Root directory for all saved artefacts.
        noise_levels: Noise levels for noisy rollout generation (Step 3).
        n_rollouts_per_noise: Rollouts collected per noise level.
        n_ensemble: Number of reward networks in the ensemble.
        rl_total_timesteps: Total timesteps for final RL training (Step 6).
        run_individual_steps: Subset of step numbers (1–6) to execute.
            Defaults to all six steps.
        retrain: Steps to force-retrain even when cached artefacts exist.
            Accepts ``None`` (use cache), ``"all"``, or a list of names from
            ``("bc", "mpc", "rollouts", "ranking", "irl", "rl")``.
        variant_kwargs: Extra keyword arguments forwarded to
            :class:`VariantNTRILTrainer` (e.g. ``ranked_data_source``,
            ``include_mpc_step``, ``reward_hidden_sizes``).
        robust_mpc: Robust Tube MPC instance required for Steps 2 and 3.
        reference_trajectory: Global sinusoidal reference numpy array injected
            into the environment options.

    Returns:
        Trained :class:`VariantNTRILTrainer` instance.
    """
    variant_kwargs = dict(variant_kwargs or {})
    env_options = dict(env_options or {"max_episode_seconds": 200.0, "dt": 1.0})
    env_options["reference_trajectory"] = reference_trajectory

    print("\n" + "=" * 70)
    print("NTRIL VARIANT PIPELINE — DOUBLEINTEGRATOR-V0")
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

    print("\nInitializing Variant NTRIL Trainer...")
    print(f"  Noise levels        : {tuple(noise_levels)}")
    print(f"  Rollouts per level  : {n_rollouts_per_noise}")

    common_kwargs = dict(
        venv=venv,
        custom_logger=custom_logger,
        noise_levels=tuple(noise_levels),
        n_rollouts_per_noise=n_rollouts_per_noise,
        n_ensemble=n_ensemble,
        irl_batch_size=32,
        irl_lr=1e-3,
        save_dir=save_dir,
        reference_trajectory=reference_trajectory,
        **variant_kwargs,
    )

    if suboptimal_policy is not None:
        variant_trainer = VariantNTRILTrainer.from_policy(suboptimal_policy, **common_kwargs)
    elif demonstrations is not None:
        variant_trainer = VariantNTRILTrainer.from_demonstrations(demonstrations, **common_kwargs)
    else:
        raise ValueError("Either 'suboptimal_policy' or 'demonstrations' must be provided.")

    if robust_mpc is not None:
        variant_trainer.robust_mpc = robust_mpc

    # Map step numbers → retrain step names.
    _STEP_NUM_TO_NAME = {1: "bc", 2: "mpc", 3: "rollouts", 4: "ranking", 5: "irl", 6: "rl"}

    if retrain == "all":
        _force_set = set(_STEP_NUM_TO_NAME.values())
    elif retrain is None:
        _force_set = set()
    else:
        _force_set = set(retrain)

    if run_individual_steps is None:
        run_individual_steps = [1, 2, 3, 4, 5, 6]

    irl_train_kwargs: Dict[str, Any] = {}
    rl_train_kwargs: Dict[str, Any] = {}
    step_results: Dict[str, Any] = {}

    print(f"\nRunning variant steps: {run_individual_steps}")

    for step_num in sorted(run_individual_steps):
        print(f"\n{'=' * 60}")
        print(f"Running Step {step_num}")
        print(f"{'=' * 60}")

        if step_num == 1:
            print("Step 1: Training BC policy from demonstrations...")
            bc_rollouts = variant_trainer._train_bc_policy(
                force_retrain="bc" in _force_set,
                progress_bar=True,
            )
            # step_results["bc"] = bc_stats
            # print("\nBC Training Stats:")
            # for key, value in bc_stats.items():
            #     print(f"  {key}: {value}")

        elif step_num == 2:
            if robust_mpc is None:
                raise ValueError("robust_mpc is required for Step 2 (RTMPC reference extraction).")
            print("Step 2: Extracting nominal reference trajectory via Robust Tube MPC...")
            rtmpc_trajectories = variant_trainer._extract_reference_trajectory_with_mpc(
                force_retrain="mpc" in _force_set
            )
            step_results["mpc"] = {"n_rtmpc_trajectories": len(rtmpc_trajectories)}
            print(f"\nExtracted {len(rtmpc_trajectories)} RTMPC reference trajectories.")

        elif step_num == 3:
            if robust_mpc is None:
                raise ValueError("robust_mpc is required for Step 3 (noisy rollout generation).")
            print("Step 3: Generating noisy rollouts from RTMPC trajectories...")
            noisy_rollouts = variant_trainer._generate_noisy_rollouts_from_rtmpc(
                force_retrain="rollouts" in _force_set
            )
            step_results["rollouts"] = {
                "n_noise_levels": len(variant_trainer.noise_levels),
                "n_rollouts_per_level": len(noisy_rollouts[0]) if noisy_rollouts else 0,
            }
            print(f"\nGenerated noisy rollouts for {len(variant_trainer.noise_levels)} noise levels.")

        elif step_num == 4:
            print("Step 4: Building ranked datasets...")
            variant_trainer._build_ranked_dataset(force_retrain="ranking" in _force_set)
            step_results["ranking"] = {
                f"dataset_{i}_num_samples": len(ds)
                for i, ds in enumerate(variant_trainer.ranked_datasets)
            }
            print("\nRanked datasets built.")

        elif step_num == 5:
            print("Step 5: Training reward networks via demonstration-ranked IRL...")
            variant_trainer._train_reward_network(
                force_retrain="irl" in _force_set,
                **(irl_train_kwargs or {}),
            )
            step_results["irl"] = {"n_ensemble": variant_trainer.n_ensemble}
            print("\nIRL training complete.")

        elif step_num == 6:
            print("Step 6: Training final policy via RL on ensemble reward...")
            variant_trainer._train_final_policy(
                total_timesteps=rl_total_timesteps,
                force_retrain="rl" in _force_set,
                **(rl_train_kwargs or {}),
            )
            step_results["rl"] = {"total_timesteps": rl_total_timesteps}
            print("\nFinal policy training complete.")

        else:
            raise ValueError(f"Invalid step number: {step_num}. Must be 1–6.")

        print(f"\n{'=' * 70}")
        print(f"Completed steps so far: {sorted(s for s in run_individual_steps if s <= step_num)}")
        print(f"{'=' * 70}")

    venv.close()
    return variant_trainer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated experimental NTRIL variant pipeline.",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=None)
    parser.add_argument("--n-rollouts-per-noise", type=int, default=5)
    parser.add_argument("--n-ensemble", type=int, default=3)
    parser.add_argument("--rl-total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--run-individual-steps",
        type=int,
        nargs="+",
        default=None,
        help="Subset of step numbers to run, e.g. --run-individual-steps 1 2 3",
    )
    parser.add_argument(
        "--retrain",
        nargs="*",
        default=None,
        help=(
            "Steps to force-retrain even when cached artefacts exist. "
            "Pass 'all' or names from: bc mpc rollouts ranking irl rl"
        ),
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
        default=[256, 256],
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

    trainer = run_variant_ntril_training(
        save_dir=str(save_dir),
        noise_levels=noise_levels,
        n_rollouts_per_noise=args.n_rollouts_per_noise,
        n_ensemble=args.n_ensemble,
        rl_total_timesteps=args.rl_total_timesteps,
        run_individual_steps=args.run_individual_steps,
        retrain=retrain,
        variant_kwargs=variant_kwargs,
    )

    archive_name = (
        args.archive_name
        if args.archive_name
        else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_variant"
    )
    trainer.archive_run(
        name=archive_name,
        archive_root=str(save_dir / "archived_runs"),
    )
    print(f"\nAll variant outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
