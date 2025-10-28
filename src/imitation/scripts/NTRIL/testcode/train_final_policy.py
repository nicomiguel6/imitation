"""Train final policy using learned reward network"""

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

from torch.utils.data.dataloader import DataLoader

from imitation.data import rollout
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector, NoisyPolicy
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import visualize_noise_levels, analyze_ranking_quality
from imitation.scripts.NTRIL.demonstration_ranked_irl import RankedTransitionsDataset, DemonstrationRankedIRL
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup environment
    rngs = np.random.default_rng()
    venv = util.make_vec_env("MountainCarContinuous-v0", rng=rngs, post_wrappers = [lambda e, _: RolloutInfoWrapper(e)])


    # Load reward network
    save_dir = "/home/nicomiguel/imitation/src/imitation/scripts/NTRIL/testcode/saved_reward_network"
    state_path = os.path.join(save_dir, "reward_net_state.pth")
    os.makedirs(save_dir, exist_ok=True)

    loaded_state = torch.load(state_path, map_location=device)
    reward_net = TrajectoryRewardNet(observation_space=venv.observation_space,
                                     action_space=venv.action_space)
    reward_net.load_state_dict(loaded_state)
    reward_net.eval()

    # Setup RL training
    relabel_reward_fn = functools.partial(reward_net.predict_processed, update_stats=False)

    learned_reward_venv = RewardVecEnvWrapper(venv, relabel_reward_fn)
    agent = PPO("MlpPolicy", learned_reward_venv, n_steps=2048 // learned_reward_venv.num_envs)
    agent.learn(total_timesteps=100_000)
    

    return None


if __name__=='__main__':
    main()
