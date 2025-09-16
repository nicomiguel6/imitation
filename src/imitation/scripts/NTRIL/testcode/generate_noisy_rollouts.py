"""Code for testing and debugging NTRIL. Step 3: Generating noisy rollouts."""

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
from imitation.algorithms import bc
from imitation.algorithms.bc import BC
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize

def main():

    """Load BC policy"""
    policy_path = "BC_policy"
    bc_policy = bc.reconstruct_policy(policy_path, device="cuda")

    










if __name__=="__main__":
    main()