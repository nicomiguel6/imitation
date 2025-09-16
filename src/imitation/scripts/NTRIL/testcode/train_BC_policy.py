"""Code for testing and debugging NTRIL. Step 2: Training BC from policy."""

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
from imitation.algorithms.bc import BC
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize

def main():
    
    """ Load expert trajectories as training data """
    traj_path = "expert_traj"
    expert_traj = serialize.load(traj_path)

    """ Load environment observation and action spaces """
    # Setup environment
    rngs = np.random.default_rng()
    venv = util.make_vec_env("MountainCarContinuous-v0", rng=rngs, post_wrappers = [lambda e, _: RolloutInfoWrapper(e)])
    

    """ Instantiate BC """
    bc_policy = BC(observation_space=venv.observation_space, 
                    action_space=venv.action_space, 
                    rng=rngs,
                    demonstrations=expert_traj,
                    device="cuda")
    
    """ Train BC Policy """
    bc_policy.train(n_epochs=10)

    """ Save BC Policy """
    policy_path = "BC_policy"
    util.save_policy(bc_policy.policy, policy_path)
    

if __name__ == "__main__":
    main()