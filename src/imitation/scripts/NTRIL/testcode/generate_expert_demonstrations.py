"""Code for testing and debugging NTRIL. Step 1: Generating expert policies."""

import numpy as np
import gymnasium as gym
import osqp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import visualize_noise_levels, analyze_ranking_quality
from imitation.util.logger import configure

def main():
    """Generate expert demonstration on Mountain Car Continuous."""
    print("Generating expert demonstrations on MountainCarContinuous-v0...")
    
    # Setup environment
    def make_env():
        return gym.make("MountainCarContinuous-v0")
    
    venv = DummyVecEnv([make_env])
    
    # Train expert policy (or load pre-trained)
    print("Training expert policy...")
    expert_policy = PPO("MlpPolicy", venv, verbose=0)
    expert_policy.learn(total_timesteps=10000)
    
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    expert_trajectories = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=10),
    )
    
    print(f"Generated {len(expert_trajectories)} expert trajectories")