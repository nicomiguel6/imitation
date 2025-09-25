"""Code for testing and debugging NTRIL. Step 3: Robust Tube MPC."""

import numpy as np
import os
import gymnasium as gym
import osqp
import hypothesis
import hypothesis.strategies as st
import casadi as ca
import do_mpc
import cdd

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

    # Set up disturbance with cdd
    W_vertex = np.array([[0.15, 0.15],[0.15, -0.15],[-0.15,-0.15],[-0.15,0.15]], dtype=object)
    gen_mat = cdd.Matrix(
        np.hstack([np.ones((W_vertex.shape[0], 1), dtype=object), W_vertex]).tolist(),
        number_type='fraction'
    )
    gen_mat.rep_type = cdd.RepType.GENERATOR

    W = cdd.Polyhedron(gen_mat)

    # Set up discrete disturbed linear system
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    

    
    return None


if __name__ == "__main__":
    main()
