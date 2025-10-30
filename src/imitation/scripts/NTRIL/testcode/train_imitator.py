"""Train Imitator using ranked dataset"""

import numpy as np
import torch
import os
import seals
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

def main():

    # load training data
    save_path_traj = "/home/nicomiguel/imitation/src/imitation/scripts/NTRIL/testcode/trajectories"
    save_path_data = "/home/nicomiguel/imitation/src/imitation/scripts/NTRIL/testcode/training_data.pth"

    loaded_traj = serialize.load(save_path_traj)
    loaded_data = torch.load(save_path_data)

    # Set up Dataset
    training_dataset = RankedTransitionsDataset(demonstrations = [loaded_traj],
                                                num_snippets = loaded_data['num_snippets'],
                                                min_segment_length = loaded_data['min_segment_length'],
                                                max_segment_length = loaded_data['max_segment_length'],
                                                generate_samples = False)
    

    training_dataset.load_training_samples(loaded_data['samples'])

    # Set up DataLoader and custom collate function
    def my_collate(batch):
        segment_data = [batch_data[0] for batch_data in batch]
        labels = [batch_data[1] for batch_data in batch]
        return segment_data, labels
    batch_size = 1
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)

    # Setup environment
    rngs = np.random.default_rng()
    venv = util.make_vec_env("MountainCarContinuous-v0", rng=rngs, post_wrappers = [lambda e, _: RolloutInfoWrapper(e)])

    # Set up reward network
    obs_space = venv.observation_space
    act_space = venv.action_space
    reward_net = TrajectoryRewardNet(observation_space=obs_space,
                                action_space=act_space,
                                use_state=True,
                                use_action=False,
                                use_next_state=False,
                                use_done=False,
                                hid_sizes=(256,256),
                           )                              
    
    

    reward_learner = DemonstrationRankedIRL(reward_net=reward_net,
                                            venv=venv,
                                            batch_size=batch_size,
                                            )


    
    reward_learner.train(train_dataloader=train_dataloader)

    # test

    # save trained reward network
    save_dir = "/home/nicomiguel/imitation/src/imitation/scripts/NTRIL/testcode/saved_reward_network"
    os.makedirs(save_dir, exist_ok=True)

    # Save state_dict (recommended)
    state_path = os.path.join(save_dir, "reward_net_state.pth")
    torch.save(reward_net.state_dict(), state_path)

    # Optionally save the whole model (less portable but convenient)
    full_path = os.path.join(save_dir, "reward_net_full.pt")
    torch.save(reward_net, full_path)

    # Save minimal metadata to help reload (env spaces, hyperparams, etc.)
    meta = {
        "obs_space": venv.observation_space,
        "act_space": venv.action_space,
        "use_state": True,
        "use_action": False,
        "use_next_state": False,
        "use_done": False,
        "hid_sizes": (256, 256),
    }
    torch.save(meta, os.path.join(save_dir, "reward_net_meta.pth"))

    # Example (commented) how to reload the state_dict later:
    # loaded_state = torch.load(state_path, map_location="cpu")
    # reward_net.load_state_dict(loaded_state)

    return None

if __name__=="__main__":
    main()