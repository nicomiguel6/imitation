"""Evaluate final policy"""

import functools
import numpy as np
import torch
import os
import gymnasium as gym
import osqp
import hypothesis
import hypothesis.strategies as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pathlib

from torch.utils.data.dataloader import DataLoader

from imitation.data import rollout
from imitation.scripts.NTRIL.noise_injection import (
    EpsilonGreedyNoiseInjector,
    NoisyPolicy,
)
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import (
    visualize_noise_levels,
    analyze_ranking_quality,
)
from imitation.scripts.NTRIL.demonstration_ranked_irl import (
    RankedTransitionsDataset,
    DemonstrationRankedIRL,
)
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize
from imitation.data import types
from imitation.algorithms import bc
from imitation.algorithms.bc import BC
from imitation.rewards.reward_nets import TrajectoryRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.scripts.train_rl import train_rl


def main():
    # reload device
    device = "mps"
    if device == "mps":
        torch.set_default_dtype(torch.float32)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    save_path_traj = SCRIPT_DIR / "trajectories"
    save_path_data = SCRIPT_DIR / "training_data.pth"

    # Setup environment
    rngs = np.random.default_rng()
    venv = util.make_vec_env(
        "seals/MountainCar-v0",
        rng=rngs,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Load saved final policy
    save_dir = SCRIPT_DIR / "trained_reward_net"
    policy_path = save_dir / "final_policy.zip"

    final_policy = PPO.load(policy_path, device=device)
    print(f"Final policy loaded from {policy_path}")

    # Evaluate on environment
    num_eval_episodes = 10
    rewards = []
    for episode in range(num_eval_episodes):
        obs = venv.reset()
        done = [False] * venv.num_envs
        done = np.array(done)
        episode_reward = 0.0

        while not done.any():
            action, _states = final_policy.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    return None


if __name__ == "__main__":
    main()
