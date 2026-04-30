import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import PPO


from imitation.scripts.SSRR.curve_fit import estimate_airl_returns_by_noise, fit_sigmoid_noise_performance, _sigmoid
from imitation.scripts.SSRR.types import NoisePerformanceData
from imitation.util import util
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.SSRR.noise_rollouts import generate_noisy_rollout_buckets
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


def test_sigmoid_fit_high_r2_on_synthetic():
    # Synthetic decreasing sigmoid-like curve with noise.
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 21)
    # True params: x0=0.5, y0=0.05, c=0.95, k=-8 (decreasing with noise)
    y_true = 0.95 / (1.0 + np.exp(-(-8.0) * (x - 0.5))) + 0.05
    y_noisy = y_true + rng.normal(scale=0.01, size=y_true.shape)

    data = NoisePerformanceData(
        noise_levels=x,
        returns_mean=y_noisy,
        returns_std=np.zeros_like(y_noisy),
        returns_all=None,
    )
    params, diag = fit_sigmoid_noise_performance(data, normalize_y=True, prefer_scipy=True)

    data = [x, y_noisy, y_true]

    assert np.isfinite(diag.r2)
    assert diag.r2 > 0.90
    # Should be monotone decreasing over [0,1] with negative k (common case).
    assert params.k < 0.0

    return params, diag, data

if __name__ == "__main__":
    ''' Test sigmoid fit on noisy trajectories generated from AIRL policy trained on double integrator.
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
    force_generate_noisy_trajectories = True
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

    # Fit sigmoid
    noise_performance_data = estimate_airl_returns_by_noise(buckets = noisy_trajectories, airl_reward = airl_reward_net)
    sigmoid_params, diag = fit_sigmoid_noise_performance(noise_performance_data, normalize_y=True, prefer_scipy=True)
    print(sigmoid_params)

    # Save sigmoid params
    np.save(airl_run_dir / "sigmoid_params.npy", np.asarray(sigmoid_params.as_tuple(), dtype=np.float64))

    # Plot sigmoid fit
    independent_noise_levels = np.linspace(0, 1.1, 1500)
    predicted_returns = _sigmoid(sigmoid_params, independent_noise_levels)

    total_noise_levels = []
    
    for bucket in noisy_trajectories:
        for traj in bucket.trajectories:
            total_noise_levels.append(traj.infos[0]['noise_level'])

    returns_all = noise_performance_data.returns_all
    
    # Normalize returns_all to be between 0 and 1
    returns_all = (returns_all - np.min(returns_all)) / (np.max(returns_all) - np.min(returns_all))

    plt.figure()
    plt.scatter(total_noise_levels, returns_all, c='r', marker='.', label='Raw Returns')
    plt.plot(independent_noise_levels, predicted_returns, 'b-', label='Fitted Returns')
    plt.xlabel('Noise Level')
    plt.ylabel('Normalized Return')
    plt.legend()

    plt.savefig(airl_run_dir / "sigmoid_fit.png")
    plt.close()

    

