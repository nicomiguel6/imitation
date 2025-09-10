"""Code for testing and debugging NTRIL. Step 1: Generating expert policies."""

import numpy as np
import os
import gymnasium as gym
import osqp
import hypothesis
import hypothesis.strategies as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import visualize_noise_levels, analyze_ranking_quality
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize

def main():
    """Generate expert demonstration on Mountain Car Continuous."""
    print("Generating expert demonstrations on MountainCarContinuous-v0...")

    # Path to save/load the model
    model_path = "expert_policy.zip"
    

    rngs = np.random.default_rng()
    
    # Setup environment
    venv = util.make_vec_env("MountainCarContinuous-v0", rng=rngs, post_wrappers = [lambda e, _: RolloutInfoWrapper(e)])
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        expert_policy = PPO.load(model_path, env=venv)
    else:
        # Train expert policy (or load pre-trained)
        print("Training expert policy...")
        expert_policy = PPO("MlpPolicy", venv, verbose=0)
        expert_policy.learn(total_timesteps=10000)

        # Save trained model
        expert_policy.save(model_path)

    
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    expert_trajectories = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=10),
        rng=rngs
    )
    
    print(f"Generated {len(expert_trajectories)} expert trajectories")
    # Save expert trajectories
    traj_path = "expert_traj"
    serialize.save(traj_path, expert_trajectories)
    print(f"Expert trajectories saved to {traj_path}")

if __name__ == "__main__":
    main()
