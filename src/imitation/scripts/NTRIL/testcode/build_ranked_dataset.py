"""Code for testing and debugging NTRIL. Step 4: Build ranked dataset."""

import numpy as np
import os
import seals
import gymnasium as gym
import osqp
import hypothesis
import hypothesis.strategies as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from imitation.data import rollout
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector, NoisyPolicy
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import visualize_noise_levels, analyze_ranking_quality
from imitation.scripts.NTRIL.demonstration_ranked_irl import RankedTransitionsDataset
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize
from imitation.data import types
from imitation.algorithms import bc
from imitation.algorithms.bc import BC


def main():
    """ Build dataset of ranked demonstration data"""

    # Import sample trajectories
    """Load BC policy"""
    policy_path = "/home/nicomiguel/imitation/src/imitation/scripts/NTRIL/testcode/BC_policy"
    bc_policy = bc.reconstruct_policy(policy_path, device="cuda")

    """Noise Injector"""
    epsilon = 0.5
    injector = EpsilonGreedyNoiseInjector()

    """Rollout trajectory"""
    noisy_bc_policy = injector.inject_noise(bc_policy, noise_level=epsilon)

    """Generate environment"""
    rngs = np.random.default_rng()
    
    # Setup environment
    venv = util.make_vec_env("seals/MountainCarContinuous-v0", rng=rngs, post_wrappers = [lambda e, _: RolloutInfoWrapper(e)])
    
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    expert_trajectories = rollout.rollout(
        noisy_bc_policy,
        venv,
        rollout.make_sample_until(min_episodes=10),
        rng=rngs,
        exclude_infos=False,
        label_info={'noise_level': epsilon},
    )
    epsilon = 0.1
    injector = EpsilonGreedyNoiseInjector()

    noisy_bc_policy_2 = injector.inject_noise(bc_policy, noise_level=epsilon)

    expert_trajectories_additional = rollout.rollout(
        noisy_bc_policy_2,
        venv,
        rollout.make_sample_until(min_episodes=10),
        rng=rngs,
        exclude_infos=False,
        label_info={'noise_level': epsilon},
    )

    # Setup ranking dictionary
    test_dataset = RankedTransitionsDataset([expert_trajectories, expert_trajectories_additional])
    


    print(len(test_dataset.demonstrations))

    training_data = test_dataset.training_data

    ## Store trajectories
    save_path_traj = "/home/nicomiguel/imitation/src/imitation/scripts/NTRIL/testcode/trajectories"
    serialize.save(save_path_traj, test_dataset.demonstrations)

    ## Store training data
    save_path_data = "/home/nicomiguel/imitation/src/imitation/scripts/NTRIL/testcode/training_data.pth"
    training_samples = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        training_samples.append(sample)
    
    torch.save({'samples': training_samples, 
                'num_snippets': test_dataset.num_snippets,
                'min_segment_length': test_dataset.min_segment_length,
                'max_segment_length': test_dataset.max_segment_length,
                }, save_path_data)


    return None



if __name__=="__main__":
    main()




    


    