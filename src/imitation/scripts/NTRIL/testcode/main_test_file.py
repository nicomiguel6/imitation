"""Main test file to run through an entire example training run."""

import pathlib
import numpy as np
import torch
import os
from pathlib import Path
import seals
import gymnasium as gym
import osqp
import hypothesis
import hypothesis.strategies as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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


print("Generating expert demonstrations on seals' MountainCar-v0...")

# Get the directory where THIS script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Define all paths relative to script directory
DEBUG_DIR = SCRIPT_DIR / "debug"
MODEL_PATH = DEBUG_DIR / "test_expert_policy.zip"
INITIAL_DEMO_PATH = DEBUG_DIR / "initial_expert_demos"

device = "cuda"
if device == "mps":
    torch.set_default_dtype(torch.float32)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rngs = np.random.default_rng()

# Setup environment
venv = util.make_vec_env(
    "seals/MountainCar-v0",
    rng=rngs,
    post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
)

if MODEL_PATH.exists():
    print(f"Loading existing model from {MODEL_PATH}...")
    expert_policy = PPO.load(MODEL_PATH, env=venv, device=device)
else:
    # Train expert policy (or load pre-trained)
    print("Training expert policy...")
    expert_policy = PPO("MlpPolicy", venv, verbose=0, device=device)
    expert_policy.learn(total_timesteps=10_000)

    # Save trained model
    expert_policy.save(MODEL_PATH)


if INITIAL_DEMO_PATH.exists():
    print(f"Loading existing expert demonstrations from {INITIAL_DEMO_PATH}...")
    expert_trajectories = serialize.load(INITIAL_DEMO_PATH)
    print(f"Loaded {len(expert_trajectories)} expert trajectories")
else:
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    expert_trajectories = rollout.rollout(
        expert_policy, venv, rollout.make_sample_until(min_episodes=100), rng=rngs
    )
    serialize.save(INITIAL_DEMO_PATH, expert_trajectories)
    print(f"Generated {len(expert_trajectories)} expert trajectories")


# Setup NTRIL trainer
base_trainer = NTRILTrainer(demonstrations=expert_trajectories,
                            venv=venv,
                            save_dir=str(DEBUG_DIR))

a = 5


