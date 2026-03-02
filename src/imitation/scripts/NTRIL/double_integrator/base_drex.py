"""DREX Baseline for the Double Integrator System.

Implements DREX (Demonstration Ranked Reward Extrapolation) *without* MPC
data augmentation, serving as a comparison baseline for NTRIL.

Pipeline (NTRIL steps 1, 2, 4, 5, 6 — step 3 omitted):
  Step 1: Set up suboptimal policy (epsilon-greedy PID controller)
  Step 2: Generate noisy rollouts at varying noise levels
  Step 4: Build n_ensemble ranked datasets directly from noisy rollouts
  Step 5: Train n_ensemble reward networks via demonstration-ranked IRL
  Step 6: Train final policy via RL using the ensemble-averaged reward

The only structural difference from NTRILTrainer is the data source for
Step 4: DREX feeds noisy rollouts directly into the ranked dataset, whereas
NTRILTrainer first augments those rollouts with the robust tube MPC (Step 3).
All ensemble training logic lives in the shared NTRILTrainer base class.
"""

import dataclasses
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.scripts.NTRIL.double_integrator.double_integrator import (
    DoubleIntegratorSuboptimalPolicy,
)
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.util import logger as imit_logger
from imitation.util import util

import gymnasium as gym


@dataclasses.dataclass
class DREXTrainer(NTRILTrainer):
    """DREX baseline trainer.

    Inherits all ensemble training logic from :class:`NTRILTrainer`.  The
    only override is :meth:`_get_ranked_dataset_source`, which returns
    ``self.noisy_rollouts`` instead of MPC-augmented data so that Step 3
    (robust tube MPC) is bypassed.

    The ``train()`` method is also overridden to skip Step 3 in the
    pipeline loop.
    """

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        irl_train_kwargs: Optional[Mapping[str, Any]] = None,
        rl_train_kwargs: Optional[Mapping[str, Any]] = None,
        retrain: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, Any]:
        """Run the DREX pipeline (Steps 1, 2, 4, 5, 6 — Step 3 omitted).

        Args:
            total_timesteps: Total timesteps for final RL training.
            bc_train_kwargs: Extra kwargs forwarded to BC training.
            irl_train_kwargs: Extra kwargs forwarded to IRL training.
            rl_train_kwargs: Extra kwargs forwarded to RL training.
            retrain: Which steps to force-retrain even when cached artefacts
                exist.  Accepts ``None`` (use cache), ``"all"``, or a list of
                names from ``("bc", "rollouts", "ranking", "irl", "rl")``.

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

        self._logger.log("Step 1: Obtaining suboptimal policy...")
        stats["bc"] = self._train_bc_policy(
            force_retrain="bc" in force, **(bc_train_kwargs or {})
        )

        self._logger.log("Step 2: Generating noisy rollouts...")
        _, rollout_stats = self._generate_noisy_rollouts(
            force_retrain="rollouts" in force
        )
        stats["rollouts"] = rollout_stats

        self._logger.log(
            f"Step 4: Building {self.n_ensemble} ranked datasets from noisy rollouts..."
        )
        self._build_ranked_dataset(force_retrain="ranking" in force)
        stats["ranking"] = {
            f"dataset_{i}_num_samples": len(ds)
            for i, ds in enumerate(self.ranked_datasets)
        }

        self._logger.log(
            f"Step 5: Training {self.n_ensemble} reward networks via IRL..."
        )
        self._train_reward_network(
            force_retrain="irl" in force, **(irl_train_kwargs or {})
        )
        stats["irl"] = {"n_ensemble": self.n_ensemble}

        self._logger.log("Step 6: Training final policy via RL (ensemble reward)...")
        self._train_final_policy(
            total_timesteps=total_timesteps,
            force_retrain="rl" in force,
            **(rl_train_kwargs or {}),
        )
        stats["rl"] = {"total_timesteps": total_timesteps}

        return stats

    # ------------------------------------------------------------------
    # Data source override
    # ------------------------------------------------------------------

    def _get_ranked_dataset_source(self) -> list:
        """Return noisy rollouts as the ranked-dataset data source.

        Overrides the parent, which uses MPC-augmented trajectories.
        """
        if not self.noisy_rollouts:
            noisy_rollouts_path = os.path.join(self.save_dir, "noisy_rollouts.pkl")
            if os.path.exists(noisy_rollouts_path):
                with open(noisy_rollouts_path, "rb") as f:
                    self.noisy_rollouts = pickle.load(f)
            else:
                raise ValueError(
                    "Noisy rollouts not found. Run _generate_noisy_rollouts() first."
                )
        return self.noisy_rollouts


# ---------------------------------------------------------------------------
# Convenience entry-points
# ---------------------------------------------------------------------------

def run_drex_training(
    env_id: str = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0",
    save_dir: str = "./drex_outputs",
    noise_levels: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    n_rollouts_per_noise: int = 10,
    n_ensemble: int = 3,
    rl_total_timesteps: int = 1_000_000,
    retrain: Optional[Union[str, Sequence[str]]] = None,
) -> DREXTrainer:
    """Set up and run the DREX baseline pipeline on the Double Integrator.

    Args:
        env_id: Gymnasium environment ID.
        save_dir: Directory for all saved artefacts.
        noise_levels: Epsilon-greedy noise levels for rollout generation.
        n_rollouts_per_noise: Number of rollouts collected per noise level.
        n_ensemble: Number of reward networks in the ensemble.
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
    print(f"Ensemble    : {n_ensemble} reward networks")
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
        n_ensemble=n_ensemble,
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
        noise_levels=tuple(np.arange(0.0, 1.0, 0.05)),
        n_rollouts_per_noise=5,
        n_ensemble=3,
        rl_total_timesteps=1_000_000,
        retrain=None,
    )

    print(f"\nAll outputs saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
