"""Code for testing and debugging NTRIL. Step 3: Robust Tube MPC."""

import numpy as np
import os
import gymnasium as gym
import osqp
import hypothesis
import hypothesis.strategies as st
import casadi as ca
import do_mpc
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

    # Set up discrete linear system
    A = np.array([[1,1],[0,1]])
    B = np.array([[0.5], [1]])
    Q = np.diag([1,1])
    R = 0.1

    W_vertex = np.array([0.15, 0.15],[0.15, -0.15],[-0.15,-0.15],[-0.15,0.15])
    W = Polyhedron(W_vertex)

    

    
    return None


if __name__ == "__main__":
    main()
