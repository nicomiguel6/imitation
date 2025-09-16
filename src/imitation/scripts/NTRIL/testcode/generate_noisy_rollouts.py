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
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector, NoisyPolicy
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

    """Noise Injector"""
    epsilon = 0.5
    injector = EpsilonGreedyNoiseInjector()

    """Rollout trajectory"""
    noisy_bc_policy = injector.inject_noise(bc_policy, noise_level=epsilon)

    """Generate environment"""
    rngs = np.random.default_rng()
    
    # Setup environment
    venv = util.make_vec_env("MountainCarContinuous-v0", rng=rngs, post_wrappers = [lambda e, _: RolloutInfoWrapper(e)])
    
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    expert_trajectories = rollout.rollout(
        noisy_bc_policy,
        venv,
        rollout.make_sample_until(min_episodes=10),
        rng=rngs
    )



if __name__=="__main__":
    main()