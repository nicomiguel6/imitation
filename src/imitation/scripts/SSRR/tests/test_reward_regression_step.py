import os
import pickle
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import PPO

from imitation.rewards.reward_nets import BasicRewardNet
from imitation.scripts.SSRR.reward_regression import SSRRRegressor, SnippetDataset, make_dataloader
from imitation.scripts.SSRR.curve_fit import estimate_airl_returns_by_noise, fit_sigmoid_noise_performance, _sigmoid
from imitation.scripts.SSRR.noise_rollouts import generate_noisy_rollout_buckets
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util
from imitation.util.networks import RunningNorm
from imitation.scripts.SSRR.types import SigmoidParams, SSRRRegressionConfig



def test_reward_regression_loss_decreases_simple():
    # Create a tiny reward net and a synthetic batch of snippets where the
    # correct return is 0.0 (all-zero observations/actions).
    obs_dim = 4
    act_dim = 1

    obs_space_low = -np.ones(obs_dim, dtype=np.float32)
    obs_space_high = np.ones(obs_dim, dtype=np.float32)
    act_space_low = -np.ones(act_dim, dtype=np.float32)
    act_space_high = np.ones(act_dim, dtype=np.float32)

    obs_space = gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float32)
    act_space = gym.spaces.Box(low=act_space_low, high=act_space_high, dtype=np.float32)

    net = BasicRewardNet(obs_space, act_space, hid_sizes=(16, 16))
    reg = SSRRRegressor(net, lr=1e-2, weight_decay=0.0, device="cpu")

    # Build a "dataloader-like" iterable that yields fixed batches.
    obs = th.zeros((6, obs_dim), dtype=th.float32)      # L+1=6
    acts = th.zeros((5, act_dim), dtype=th.float32)     # L=5
    targets = th.zeros((4,), dtype=th.float32)

    batch = ([obs] * 4, [acts] * 4, targets)
    dl = [batch] * 50

    losses = reg.train(dl, n_steps=50)["step_losses"]
    assert losses[-1] < losses[0]


if __name__ == "__main__":
    ''' Test reward regression step of SSRR.
    '''
    from pathlib import Path
    SCRIPT_DIR = Path(__file__).parent.resolve()

    AIRL_RUN = "20260420_231943_sinusoidal_A1.0_f0.01"
    airl_run_dir = SCRIPT_DIR / "airl_outputs" / AIRL_RUN

    # Set random generator
    rngs = np.random.default_rng(42)

    # Load reference trajectory
    reference_trajectory = np.load(airl_run_dir / "reference_trajectory.npy")

    # Environment and simulation parameters
    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"
    max_episode_seconds = 1000.0
    dt = 1.0

    # Create environment
    env_options = {
        "max_episode_seconds": max_episode_seconds,
        "dt": dt,
        "reference_trajectory": reference_trajectory,
        "disturbance_magnitude": 0.0,
    }

    venv = util.make_vec_env(
        env_id,
        rng=rngs,
        n_envs=5,
        parallel=True,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        env_make_kwargs=env_options,
    )

    # Load AIRL policy trained on double integrator and associated reward network
    airl_gen = PPO.load(airl_run_dir / "best_checkpoint" / "learner_policy")
    airl_policy = airl_gen.policy

    airl_reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_reward_weights = torch.load(airl_run_dir / "best_checkpoint" / "reward_net.pt", weights_only=True)
    airl_reward_net.load_state_dict(airl_reward_weights)


    # Check if noisy trajectories are already generated in that trained AIRL policy
    force_generate_noisy_trajectories = False
    noisy_trajectories_path = airl_run_dir / "noisy_trajectories.pkl"
    if os.path.exists(noisy_trajectories_path) and not force_generate_noisy_trajectories:
        noisy_trajectories = pickle.load(open(noisy_trajectories_path, "rb"))
    else:
        # Generate noisy trajectories
        noisy_trajectories = generate_noisy_rollout_buckets(base_policy=airl_policy, 
                                                            noise_levels=tuple(np.arange(0.0, 1.05, 0.05)),
                                                            n_rollouts_per_noise=5, rng=np.random.default_rng(0),
                                                            venv=venv,
                                                            reference_trajectory=reference_trajectory,
        )
        pickle.dump(noisy_trajectories, open(noisy_trajectories_path, "wb"))

    # Load sigmoid params
    sigmoid_params = np.load(airl_run_dir / "sigmoid_params.npy")
    sigmoid_params = SigmoidParams(x0=sigmoid_params[0], y0=sigmoid_params[1], c=sigmoid_params[2], k=sigmoid_params[3])

    # Device
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Initialize reward network
    reward_network = BasicRewardNet(observation_space=venv.observation_space, 
                                    action_space=venv.action_space, 
                                    normalize_input_layer=RunningNorm,
                                    )

    # Initialize reward regressor
    regressor = SSRRRegressor(reward_net=reward_network, lr=1e-2, weight_decay=0.0, device=device)

    # Initialize DataLoader
    dataloader = make_dataloader(buckets=noisy_trajectories, 
                                sigmoid=sigmoid_params, 
                                num_snippets=100, 
                                cfg=SSRRRegressionConfig(min_steps=10, max_steps=10, target_scale=10.0, length_normalize=True), 
                                batch_size=10, 
                                rng=np.random.default_rng(0))
    
    # Train reward regressor
    regressor.train(dataloader, n_steps=100)