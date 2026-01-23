"""Code for testing and debugging NTRIL. Step 1: Generating expert policies."""

import numpy as np
import os
import gymnasium as gym
import seals

# import osqp
# import hypothesis
# import hypothesis.strategies as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from imitation.data import rollout

# from imitation.scripts.NTRIL.ntril import NTRILTrainer
# from imitation.scripts.NTRIL.utils import (
#     visualize_noise_levels,
#     analyze_ranking_quality,
# )
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize


class SaveSuboptimalModelsCallback(BaseCallback):
    """
    Callback to save model parameters at regular intervals during training.
    This allows capturing suboptimal policies at different stages of learning.
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, f"model_step_{self.num_timesteps}.zip"
            )
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved suboptimal model at step {self.num_timesteps}")
        return True


def main():
    """Train expert policy and save checkpoints during training."""
    print("Training expert policy on seals' MountainCar-v0...")

    # Path to save the final expert model
    model_path = "expert_policy.zip"

    rngs = np.random.default_rng()

    # Setup environment
    venv = util.make_vec_env(
        "seals/MountainCar-v0",
        rng=rngs,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Train expert policy and save checkpoints
    print("Training expert policy...")
    expert_policy = PPO("MlpPolicy", venv, verbose=1)

    # Create callback to save suboptimal models during training
    suboptimal_save_path = "suboptimal_models"
    save_callback = SaveSuboptimalModelsCallback(
        save_freq=1000,  # Save every 1000 steps
        save_path=suboptimal_save_path,
        verbose=1,
    )

    expert_policy.learn(total_timesteps=10000, callback=save_callback)

    # Save final trained model
    expert_policy.save(model_path)
    print(f"Final expert policy saved to {model_path}")


if __name__ == "__main__":
    main()
