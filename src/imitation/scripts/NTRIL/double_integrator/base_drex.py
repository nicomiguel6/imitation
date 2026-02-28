"""DREX Baseline for the Double Integrator System.

Implements DREX (Demonstration Ranked Reward Extrapolation) *without* MPC
data augmentation, serving as a comparison baseline for NTRIL.

Pipeline (NTRIL steps 1, 2, 4, 5, 6 — step 3 omitted):
  Step 1: Set up suboptimal policy (epsilon-greedy PID controller)
  Step 2: Generate noisy rollouts at varying noise levels
  Step 4: Build ranked dataset directly from noisy rollouts
  Step 5: Train reward network via demonstration-ranked IRL
  Step 6: Train final policy via RL with the learned reward
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch as th

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import TrajectoryRewardNet
from imitation.scripts.NTRIL.demonstration_ranked_irl import RankedTransitionsDataset
from imitation.scripts.NTRIL.double_integrator.double_integrator import (
    DoubleIntegratorSuboptimalPolicy,
)
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.util import logger as imit_logger
from imitation.util import util

import gymnasium as gym


class DREXTrainer(NTRILTrainer):
    """DREX baseline trainer.

    Extends :class:`NTRILTrainer` but skips MPC data augmentation (Step 3),
    building the ranked dataset directly from the noisy rollouts produced in
    Step 2.  All other pipeline steps are inherited unchanged.
    """

    def train(
        self,
        total_timesteps: int,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        irl_train_kwargs: Optional[Mapping[str, Any]] = None,
        rl_train_kwargs: Optional[Mapping[str, Any]] = None,
        retrain: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, Any]:
        """Run the DREX pipeline (Steps 1, 2, 4, 5, 6; Step 3 omitted).

        Args:
            total_timesteps: Total timesteps for final RL training.
            bc_train_kwargs: Extra kwargs forwarded to BC training.
            irl_train_kwargs: Extra kwargs forwarded to IRL training.
            rl_train_kwargs: Extra kwargs forwarded to RL training.
            retrain: Which pipeline steps to force-retrain even when cached
                artefacts already exist.  Accepts ``None`` (use cache),
                ``"all"``, or a list of names from
                ``("bc", "rollouts", "ranking", "irl", "rl")``.

        Returns:
            Dictionary of per-step training statistics.
        """
        if retrain == "all":
            force = set(self.STEPS)
        elif retrain is None:
            force: set = set()
        else:
            unknown = set(retrain) - set(self.STEPS)
            if unknown:
                raise ValueError(
                    f"Unknown step name(s) in retrain: {unknown}. "
                    f"Valid names: {self.STEPS}"
                )
            force = set(retrain)

        stats: Dict[str, Any] = {}

        # Step 1: Obtain suboptimal policy (trains BC, or reuses provided policy).
        self._logger.log("Step 1: Obtaining suboptimal policy...")
        bc_stats = self._train_bc_policy(
            force_retrain="bc" in force, **(bc_train_kwargs or {})
        )
        stats["bc"] = bc_stats

        # Step 2: Generate noisy rollouts at varying epsilon-greedy noise levels.
        self._logger.log("Step 2: Generating noisy rollouts...")
        _, rollout_stats = self._generate_noisy_rollouts(
            force_retrain="rollouts" in force
        )
        stats["rollouts"] = rollout_stats

        # Step 4: Build ranked dataset directly from noisy rollouts (no MPC).
        self._logger.log("Step 4: Building ranked dataset from noisy rollouts...")
        self._build_ranked_dataset(force_retrain="ranking" in force)
        stats["ranking"] = {"num_samples": len(self.ranked_dataset)}

        # Step 5: Train reward network via demonstration-ranked IRL.
        self._logger.log("Step 5: Training reward network via IRL...")
        self._train_reward_network(
            force_retrain="irl" in force, **(irl_train_kwargs or {})
        )
        stats["irl"] = {}

        # Step 6: Train final policy via RL with the learned reward.
        self._logger.log("Step 6: Training final policy via RL...")
        self._train_final_policy(
            total_timesteps=total_timesteps,
            force_retrain="rl" in force,
            **(rl_train_kwargs or {}),
        )
        stats["rl"] = {"total_timesteps": total_timesteps}

        return stats

    def _build_ranked_dataset(self, force_retrain: bool = False) -> RankedTransitionsDataset:
        """Build ranked dataset from noisy rollouts (Step 4).

        Unlike the parent class, which uses MPC-augmented data, this method
        feeds the noisy rollouts directly into :class:`RankedTransitionsDataset`,
        skipping Step 3 entirely.

        Args:
            force_retrain: Rebuild even when a cached file exists on disk.

        Returns:
            The populated :class:`RankedTransitionsDataset`.
        """
        ranked_path = os.path.join(self.save_dir, "ranked_samples.pth")

        if os.path.exists(ranked_path) and not force_retrain:
            saved = th.load(ranked_path)
            self.ranked_dataset = RankedTransitionsDataset(
                demonstrations=None,
                training_samples=saved["samples"],
                num_snippets=saved["num_snippets"],
                min_segment_length=saved["min_segment_length"],
                max_segment_length=saved["max_segment_length"],
            )
            return self.ranked_dataset

        # Ensure noisy rollouts are available.
        if not self.noisy_rollouts:
            noisy_rollouts_path = os.path.join(self.save_dir, "noisy_rollouts.pkl")
            if os.path.exists(noisy_rollouts_path):
                with open(noisy_rollouts_path, "rb") as f:
                    self.noisy_rollouts = pickle.load(f)
            else:
                raise ValueError(
                    "Noisy rollouts not found. Run _generate_noisy_rollouts() first."
                )

        # Build dataset directly from noisy rollouts — no augmentation.
        self.ranked_dataset = RankedTransitionsDataset(
            demonstrations=self.noisy_rollouts,
            num_snippets=100,
            min_segment_length=20,
            max_segment_length=20,
        )

        training_samples = [
            self.ranked_dataset[i] for i in range(len(self.ranked_dataset))
        ]
        th.save(
            {
                "samples": training_samples,
                "num_snippets": self.ranked_dataset.num_snippets,
                "min_segment_length": self.ranked_dataset.min_segment_length,
                "max_segment_length": self.ranked_dataset.max_segment_length,
            },
            ranked_path,
        )

        return self.ranked_dataset


def run_drex_training(
    env_id: str = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0",
    save_dir: str = "./drex_outputs",
    noise_levels: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    n_rollouts_per_noise: int = 10,
    rl_total_timesteps: int = 1_000_000,
    retrain: Optional[Union[str, Sequence[str]]] = None,
) -> DREXTrainer:
    """Set up and run the DREX baseline pipeline on the Double Integrator.

    Args:
        env_id: Gymnasium environment ID.
        save_dir: Directory for all saved artefacts.
        noise_levels: Epsilon-greedy noise levels for rollout generation.
        n_rollouts_per_noise: Number of rollouts collected per noise level.
        rl_total_timesteps: Total timesteps for the final RL training step.
        retrain: Steps to force-retrain (``None``, ``"all"``, or a list of
            names from ``("bc", "rollouts", "ranking", "irl", "rl")``).

    Returns:
        Trained :class:`DREXTrainer` instance.
    """
    print("\n" + "=" * 70)
    print("DREX BASELINE PIPELINE — DOUBLEINTEGRATOR-V0")
    print("=" * 70)
    print(f"\nEnvironment : {env_id}")
    print(f"Noise levels: {noise_levels}")
    print(f"Save dir    : {save_dir}")

    rng = np.random.default_rng(42)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    custom_logger = imit_logger.configure(
        folder=os.path.join(save_dir, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    reward_net = TrajectoryRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True,
        use_action=False,
        use_next_state=False,
        use_done=False,
        hid_sizes=(256, 256),
    )

    # Build the suboptimal PID policy for the double integrator.
    ghost_env = gym.make(env_id)
    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=ghost_env.observation_space,
        action_space=ghost_env.action_space,
    )
    suboptimal_policy.set_K_values(K_position=0.02, K_velocity=0.3)
    ghost_env.close()

    trainer = DREXTrainer.from_policy(
        policy=suboptimal_policy,
        venv=venv,
        custom_logger=custom_logger,
        noise_levels=noise_levels,
        n_rollouts_per_noise=n_rollouts_per_noise,
        reward_net=reward_net,
        irl_batch_size=32,
        irl_lr=1e-3,
        save_dir=save_dir,
    )

    print("\nStarting DREX training pipeline...")
    stats = trainer.train(
        total_timesteps=rl_total_timesteps,
        retrain=retrain,
    )

    print("\n" + "=" * 70)
    print("DREX Training Complete!")
    print("=" * 70)
    print("\nTraining Statistics:")
    for stage, stage_stats in stats.items():
        print(f"\n  {stage.upper()}:")
        for key, value in stage_stats.items():
            print(f"    {key}: {value}")

    venv.close()
    return trainer


def main():
    """Entry point: run the DREX baseline on the Double Integrator."""
    SCRIPT_DIR = Path(__file__).parent.resolve()
    SAVE_DIR = SCRIPT_DIR / "drex_outputs"

    run_drex_training(
        env_id="imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0",
        save_dir=str(SAVE_DIR),
        noise_levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        n_rollouts_per_noise=16,
        rl_total_timesteps=1_000_000,
        retrain=None,
    )

    print(f"\nAll outputs saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
