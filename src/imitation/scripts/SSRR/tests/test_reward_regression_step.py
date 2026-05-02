import os
import pickle
import numpy as np
import torch as th
import tqdm
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO, SAC

from imitation.rewards.reward_nets import BasicRewardNet
from imitation.scripts.SSRR.reward_regression import SSRRRegressor, SnippetDataset, make_dataloader
from imitation.scripts.SSRR.curve_fit import estimate_airl_returns_by_noise, fit_sigmoid_noise_performance, _sigmoid
from imitation.scripts.SSRR.noise_rollouts import generate_noisy_rollout_buckets
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util
from imitation.util.networks import RunningNorm
from imitation.scripts.SSRR.types import SigmoidParams, SSRRRegressionConfig
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.scripts.SSRR.reporting import write_rl_run_report
from imitation.scripts.SSRR.util import plot_learned_reward



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

    AIRL_RUN = "20260501_221911_constant_P0.0"
    airl_run_dir = SCRIPT_DIR / "airl_outputs" / AIRL_RUN
    # force_retrain = ["reward_regression", "rl_training"]
    force_retrain = []
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


    # Stage output directories (must already exist from test_curve_fit_sigmoid.py)
    noisy_rollouts_dir = airl_run_dir / "noisy_rollouts"
    sigmoid_fit_dir = airl_run_dir / "sigmoid_fit"
    ssrr_regression_dir = airl_run_dir / "ssrr_regression"
    ssrr_rl_dir = airl_run_dir / "ssrr_rl"
    ssrr_regression_dir.mkdir(parents=True, exist_ok=True)
    ssrr_rl_dir.mkdir(parents=True, exist_ok=True)

    # Check if noisy trajectories are already generated in that trained AIRL policy
    force_generate_noisy_trajectories = "noisy_rollouts" in force_retrain
    noisy_trajectories_path = noisy_rollouts_dir / "noisy_trajectories.pkl"
    if os.path.exists(noisy_trajectories_path) and not force_generate_noisy_trajectories:
        noisy_trajectories = pickle.load(open(noisy_trajectories_path, "rb"))
    else:
        noisy_rollouts_dir.mkdir(parents=True, exist_ok=True)
        # Generate noisy trajectories
        noisy_trajectories = generate_noisy_rollout_buckets(base_policy=airl_policy, 
                                                            noise_levels=tuple(np.arange(0.0, 1.05, 0.05)),
                                                            n_rollouts_per_noise=5, rng=np.random.default_rng(0),
                                                            venv=venv,
                                                            reference_trajectory=reference_trajectory,
        )
        pickle.dump(noisy_trajectories, open(noisy_trajectories_path, "wb"))

    # Load sigmoid params (produced by test_curve_fit_sigmoid.py) if not already loaded
    sigmoid_params_path = sigmoid_fit_dir / "sigmoid_params.npy"
    if os.path.exists(sigmoid_params_path):
        sigmoid_params = np.load(sigmoid_params_path)
    else:
        # Fit sigmoid
        noise_performance_data = estimate_airl_returns_by_noise(buckets = noisy_trajectories, airl_reward = airl_reward_net)
        sigmoid_params, diag = fit_sigmoid_noise_performance(noise_performance_data, normalize_y=True, prefer_scipy=True)
        np.save(sigmoid_params_path, np.asarray(sigmoid_params.as_tuple(), dtype=np.float64))

    sigmoid_params = SigmoidParams(x0=sigmoid_params[0], y0=sigmoid_params[1], c=sigmoid_params[2], k=sigmoid_params[3])

    # Device
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Train/load an ensemble of SSRR reward regressors (mirrors NTRILTrainer pattern).
    n_ensemble = 3
    ensemble_dir = ssrr_regression_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    ensemble_reward_paths = [ensemble_dir / f"ssrr_reward_{i}.pt" for i in range(n_ensemble)]
    all_cached = all(os.path.exists(path) for path in ensemble_reward_paths)

    # Regression hyperparameters aligned with NTRILTrainer._train_reward_network defaults.
    reg_batch_size = 64
    reg_lr = 1e-4
    reg_weight_decay = 0.01
    reg_n_steps = 3_000
    reg_num_snippets = 5000

    # SSRRRegressionConfig controls snippet sampling and target scaling.
    reg_min_steps = 50
    reg_max_steps = 500
    reg_target_scale = 10.0
    reg_length_normalize = True
    reg_cfg = SSRRRegressionConfig(
        min_steps=reg_min_steps,
        max_steps=reg_max_steps,
        target_scale=reg_target_scale,
        length_normalize=reg_length_normalize,
    )

    # Reward network initialization
    reward_net_init = {
        "custom_init": True}

    force_reward_regression_train = "reward_regression" in force_retrain
    reward_nets_ensemble = []
    if all_cached and not force_reward_regression_train:
        for path in ensemble_reward_paths:
            reward_network = BasicRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                normalize_input_layer=RunningNorm,
            )
            reward_network.load_state_dict(th.load(path, map_location=device))
            reward_network.to(device)
            reward_nets_ensemble.append(reward_network)
    else:
        for i in range(n_ensemble):
            print(f"\n[Ensemble {i + 1}/{n_ensemble}] Training SSRR reward regressor...")
            reward_network = BasicRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                normalize_input_layer=RunningNorm,
            )
            regressor = SSRRRegressor(
                reward_net=reward_network,
                lr=reg_lr,
                weight_decay=reg_weight_decay,
                device=device,
            )
            dataloader = make_dataloader(
                buckets=noisy_trajectories,
                sigmoid=sigmoid_params,
                num_snippets=reg_num_snippets,
                cfg=reg_cfg,
                batch_size=reg_batch_size,
                rng=np.random.default_rng(i),
            )
            regressor.train(dataloader, n_steps=reg_n_steps, log_interval=100)
            th.save(regressor.reward_net.state_dict(), ensemble_reward_paths[i])
            reward_nets_ensemble.append(regressor.reward_net)

    # Plot the mean learned reward surface over (position, velocity).
    # Mirrors the contour visualisation in NTRIL/double_integrator/util.py.
    plot_learned_reward(
        ensemble=reward_nets_ensemble,
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        ref_pos=0.0,
        ref_vel=0.0,
        n_grid=100,
        device=device,
        reference_trajectory=reference_trajectory,
        out_path=ssrr_regression_dir / "reward_contour.png",
        title=f"SSRR mean learned reward  |  run={AIRL_RUN}",
    )

    # Train RL policy on SSRR reward
    def ensemble_reward_fn(obs, acts, next_obs, dones):
        return np.mean(
            [
                net.predict_processed(obs, acts, next_obs, dones, update_stats=False)
                for net in reward_nets_ensemble
            ],
            axis=0,
        )

    # SAC hyperparameters
    rl_learning_rate = 3e-4
    rl_learning_starts = 10_000
    rl_buffer_size = 1_000_000
    rl_batch_size = 256
    rl_tau = 0.01
    rl_gamma = 0.99
    rl_train_freq = 1
    rl_gradient_steps = 1
    rl_total_timesteps = int(1e5)

    learned_reward_venv = RewardVecEnvWrapper(venv, reward_fn=ensemble_reward_fn)
    rl_policy = SAC(
        "MlpPolicy",
        learned_reward_venv,
        learning_rate=rl_learning_rate,
        learning_starts=rl_learning_starts,
        buffer_size=rl_buffer_size,
        batch_size=rl_batch_size,
        tau=rl_tau,
        gamma=rl_gamma,
        train_freq=rl_train_freq,
        gradient_steps=rl_gradient_steps,
        verbose=0,
        device=device,
    )

    # Use a custom callback to update tqdm
    class TqdmCallback:
        def __init__(self, total_timesteps):
            self.pbar = tqdm.tqdm(total=total_timesteps, desc="RL Training")
            self.last_steps = 0

        def __call__(self, locals_, globals_):
            # Called every rollout
            num_timesteps = locals_['self'].num_timesteps
            steps = num_timesteps - self.last_steps
            if steps > 0:
                self.pbar.update(steps)
                self.last_steps = num_timesteps
            if num_timesteps >= self.pbar.total:
                self.pbar.close()
            return True

    tqdm_callback = TqdmCallback(total_timesteps=rl_total_timesteps)
    rl_policy_path = ssrr_rl_dir / "ssrr_rl_policy.zip"
    # Save RL policy under a datetime-stamped subdirectory
    rl_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    rl_run_dir = ssrr_rl_dir / rl_run_name
    rl_run_dir.mkdir(parents=True, exist_ok=True)
    if os.path.exists(rl_policy_path) and "rl_training" not in force_retrain:
        rl_policy = SAC.load(rl_policy_path, device=device)
    else:
        rl_policy.learn(total_timesteps=rl_total_timesteps, log_interval=10, callback=tqdm_callback)
        rl_policy.save(rl_run_dir / "ssrr_rl_policy")
        # also save as latest
        rl_policy.save(rl_policy_path)


    write_rl_run_report(
        run_dir=rl_run_dir,
        run_name=rl_run_name,
        airl_run=AIRL_RUN,
        env_id=env_id,
        max_episode_seconds=max_episode_seconds,
        dt=dt,
        n_envs=5,
        n_ensemble=n_ensemble,
        reg_lr=reg_lr,
        reg_weight_decay=reg_weight_decay,
        reg_batch_size=reg_batch_size,
        reg_n_steps=reg_n_steps,
        reg_num_snippets=reg_num_snippets,
        reg_min_steps=reg_min_steps,
        reg_max_steps=reg_max_steps,
        reg_target_scale=reg_target_scale,
        reg_length_normalize=reg_length_normalize,
        rl_learning_rate=rl_learning_rate,
        rl_learning_starts=rl_learning_starts,
        rl_buffer_size=rl_buffer_size,
        rl_batch_size=rl_batch_size,
        rl_tau=rl_tau,
        rl_gamma=rl_gamma,
        rl_train_freq=rl_train_freq,
        rl_gradient_steps=rl_gradient_steps,
        rl_total_timesteps=rl_total_timesteps,
        device=device,
    )