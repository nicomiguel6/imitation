"""
Utility functions for the double integrator example. Allows me tto grab a trained policy and evaluate without running main_sim.py.

Author: Nicolas Miguel
Date: January 2026
"""

import os
from pathlib import Path
import functools
from typing import Optional, Tuple

import numpy as np
import torch as th
import gymnasium as gym
import dataclasses
from stable_baselines3 import PPO

from imitation.algorithms import bc
from imitation.data import rollout, serialize
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util, logger as imit_logger
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC, RobustTubeMPCPolicy

from imitation.scripts.NTRIL.testcode.apply_robust_tube_mpc import main as apply_robust_tube_mpc

import matplotlib.pyplot as plt


def test_trained_policy(policy, env, n_episodes=10):
    """Test a trained policy on a given environment."""
    returns = []
    lengths = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)
    return returns, lengths

def plot_phase_portrait(states, title) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the phase portrait of a given states."""
    fig, ax = plt.subplots()
    ax.plot(states[:, 0], states[:, 1], "k-", label="trajectory")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    ax.grid(True)
    return fig, ax

if __name__ == "__main__":
    env_policy = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0")
    env_mpc = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0")
    device = "cuda"
    if device == "mps":
        th.set_default_dtype(th.float32)
    else:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Paths
    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # Define all paths relative to script directory
    DEBUG_DIR = SCRIPT_DIR / "debug"
    CHECKPOINTS_DIR = DEBUG_DIR / "policy_checkpoints"
    DEMOS_DIR = DEBUG_DIR / "demonstrations"
    MPC_DEMOS_DIR = DEMOS_DIR / "mpc_demonstrations.pkl"
    BC_DEMOS_DIR = DEMOS_DIR / "bc_mpc_demonstrations.pkl"  
    BC_POLICY_PATH = CHECKPOINTS_DIR / "bc_mpc_policy.pkl"

    # Load the policy
    # policy = PPO.load(str(CHECKPOINTS_DIR / "expert_policy_final_2000000.zip"), env=env_policy, device=device)
    policy = bc.reconstruct_policy(str(BC_POLICY_PATH), device=device)

    # Test MPC on double integrator gym Env

    # Set up MPC as callable policy
    mpc_policy = RobustTubeMPC(
        horizon=10,
        time_step=1.0,
        A=np.array([[0.0, 1.0], [0.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        Q=np.diag([10.0, 1.0]),
        R=0.01*np.eye(1),
        state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds=(np.array([-50.0]), np.array([50.0])))
    
    mpc_policy.setup()
    
    for j in range(1):
        # Simulate the policy
        obs, info = env_policy.reset()
        obs_mpc, info = env_mpc.reset(state=obs)
        states_policy = [obs.copy()]
        states_mpc = [obs.copy()]
        actions_policy = []
        actions_mpc = []
        rewards_policy = []
        rewards_mpc = []
        for i in range(env_policy.unwrapped.max_episode_steps):
            action_policy, _ = policy.predict(obs) # trained BC policy
            _, action_mpc = mpc_policy.solve_mpc(obs_mpc) # normal MPC policy
            obs, reward, terminated, truncated, info = env_policy.step(action_policy)
            obs_mpc, reward_mpc, terminated_mpc, truncated_mpc, info_mpc = env_mpc.step(action_mpc)
            states_policy.append(obs)
            states_mpc.append(obs_mpc)
            actions_policy.append(action_policy)
            actions_mpc.append(action_mpc)
            rewards_policy.append(reward)
            rewards_mpc.append(reward_mpc)
            if terminated or truncated:
                break

        # Plot both phase portraits in same figure
        fig, ax = plt.subplots()
        ax.plot(np.array(states_policy)[:, 0], "k-", label="trajectory")
        ax.plot(np.array(states_mpc)[:, 0], "r-", label="MPC trajectory")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_title("Trajectory Comparison of Trained Policy and MPC")
        fig.savefig(DEMOS_DIR / f"trajectory_comparison_{j}.png")
        plt.close()

        # Print total policy cost
        print("Total policy cost: ", sum(rewards_policy))
        print("Total MPC cost: ", sum(rewards_mpc))
    
