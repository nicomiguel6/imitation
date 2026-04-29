''' Test the AIRL implementation '''

# Standard library imports
import json
import numpy as np
import matplotlib.pyplot as plt
from imitation.algorithms.adversarial.airl import AIRL
import gymnasium as gym
from pathlib import Path
import torch as th
from datetime import datetime
import os

# Third-party imports
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn
# Imitation library imports
from imitation.data import rollout, types
from imitation.data.types import Trajectory, TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards import reward_nets
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util import logger as imit_logger
from imitation.util import util
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
# Project-specific imports
from imitation.scripts.NTRIL.double_integrator.main_sim import generate_reference_trajectory
from imitation.scripts.NTRIL.double_integrator.double_integrator import DoubleIntegratorSuboptimalPolicy
from imitation.algorithms import bc


if __name__ == "__main__":
    # Configuration
    print("Testing AIRL on DoubleIntegrator-v0...")

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    SAVE_DIR = SCRIPT_DIR / "airl_outputs"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


    device = "cpu"
    # if device == "mps":
    #     th.set_default_dtype(th.float32)
    # else:
    #     device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Set random generator
    rngs = np.random.default_rng(42)

    # Environment and simulation parameters
    env_id = "imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0"
    max_episode_seconds = 1000.0
    dt = 1.0

    # Create environment
    env = gym.make(
        env_id, 
        max_episode_seconds=max_episode_seconds, 
        dt=dt
    )

    # Reference trajectory parameters
    ref_mode = "sinusoidal"
    ref_amplitude = 1.0
    ref_frequency = 0.01
    ref_phase = 0.0

    # Archive name for experiment results
    _auto_archive_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{ref_mode}_A{ref_amplitude}_f{ref_frequency}"
    )
    # Save directory using auto archive name
    save_dir = SAVE_DIR / _auto_archive_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate reference trajectory
    reference_trajectory = generate_reference_trajectory(
        T=env.max_episode_steps,
        dt=env.dt,
        mode=ref_mode,
        amplitude=ref_amplitude,
        frequency=ref_frequency,
        phase=ref_phase,
    )

    # Create a Trajectory object for AIRL demonstration interface
    reference_trajectory_mpc = Trajectory(
        obs=reference_trajectory,
        acts=np.zeros((env.max_episode_steps, 1)),
        infos=np.array([{}] * env.max_episode_steps),
        terminal=True
    )

    # Save the generated reference trajectory
    np.save(save_dir / "reference_trajectory.npy", reference_trajectory)
    print(f"Saved reference trajectory to {save_dir / 'reference_trajectory.npy'}")


    # Environment options for potential future use
    env_options = {
        "max_episode_seconds": max_episode_seconds,
        "dt": dt,
        "reference_trajectory": reference_trajectory
    }

    # ------------------------
    # Generate suboptimal policy and rollouts
    # ------------------------
    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # Set K values for suboptimal policy
    suboptimal_policy.set_K_values(K_position=0.02, K_velocity=0.3)

    # Create vectorized environment
    venv = util.make_vec_env(
        env_id,
        rng=rngs,
        n_envs=5,
        parallel=True,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        env_make_kwargs=env_options,
    )

    # Setup logger
    custom_logger = imit_logger.configure(
        folder=os.path.join(save_dir, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    # Rollout suboptimal policy
    suboptimal_rollouts = rollout.rollout(
        suboptimal_policy,
        venv,
        rollout.make_sample_until(min_episodes=5),
        rng=rngs,
        exclude_infos=True,
    )

    # Set up Learner
    # PPO parameters mapped from Ant AIRL (ant_airl.py):
    #   batch_size=10000  → n_steps=2000 × n_envs=5 = 10000 total steps per rollout
    #   max_path_length=1000 → episode length of 1000 steps (set via env)
    #   discount=0.99     → gamma=0.99
    #   gae_lambda=0.97   → gae_lambda=0.97
    #   entropy_weight=0.1 → ent_coef=0.1
    #   hidden_sizes=(32,32) → net_arch=[32, 32]
    #   step_size=0.01 (TRPO KL) has no direct PPO equivalent; 3e-4 is a standard PPO LR
    # PPO parameters
    n_steps = 1000
    batch_size = 64
    gamma = 0.98
    gae_lambda = 0.95
    ent_coef = 0.01
    learning_rate = get_linear_fn(3e-4, 1e-5, 1.0)
    target_kl = 0.02
    net_arch = [32, 32]
    verbose = 0
    device = "cpu"
    
    learner = PPO(env=venv, 
                    policy="MlpPolicy", 
                    n_steps=n_steps, 
                    batch_size=batch_size, 
                    gamma=gamma, 
                    gae_lambda=gae_lambda, 
                    ent_coef=ent_coef, 
                    learning_rate=learning_rate, 
                    policy_kwargs=dict(net_arch=net_arch), 
                    verbose=verbose, 
                    device=device, 
                    target_kl=target_kl,
                    max_grad_norm=0.5)

    # Warm start learner using Behavior Cloning on suboptimal rollouts.
    # Pass learner.policy directly so BC trains the PPO policy in-place,
    # avoiding any architecture mismatch when copying weights.
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=learner.policy,
        demonstrations=suboptimal_rollouts,
        rng=rngs,
        device=device,
    )
    bc_trainer.train(n_epochs=10)

    # Save the BC-warmed PPO learner in full SB3 format so it can be
    # reloaded with PPO.load(). bc_trainer.policy.save() only saves weights
    # and lacks the metadata that PPO.load() requires.
    bc_policy_path = save_dir / "initial_BC_policy"
    bc_policy_path.mkdir(parents=True, exist_ok=True)
    learner.save(bc_policy_path / "bc_policy")
    print(f"Saved initial BC policy to {bc_policy_path / 'bc_policy.zip'}")

    # Set up AIRL
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # AIRL parameters
    demo_batch_size = 512
    n_disc_updates_per_round = 2
    disc_opt_kwargs = {"lr": 1e-3}
    log_dir = save_dir
    custom_logger = custom_logger
    airl = AIRL(
        venv=venv,                          
        demonstrations=suboptimal_rollouts,
        gen_algo=learner,
        reward_net=reward_net,              
        demo_batch_size=demo_batch_size,
        n_disc_updates_per_round=n_disc_updates_per_round,
        disc_opt_kwargs=disc_opt_kwargs,
        log_dir=log_dir,
        custom_logger=custom_logger,
    )

    # Train AIRL
    venv.seed(42)
    learner_rewards_before_training, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    total_timesteps = 2_000_000

    # Checkpoint callback: saves reward net + policy whenever raw env reward hits a new best.
    # Evaluates every `ckpt_every` rounds using the unwrapped env reward (not the AIRL reward).
    best_ckpt_dir = save_dir / "best_checkpoint"
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_mean_reward = -np.inf
    ckpt_every = 10   # evaluate every 10 rounds (~50k steps with n_steps=1000, n_envs=5)

    def checkpoint_callback(round_num: int) -> None:
        global best_mean_reward
        if round_num % ckpt_every != 0:
            return
        rewards, _ = evaluate_policy(learner, venv, n_eval_episodes=10, return_episode_rewards=True)
        mean_rew = float(np.mean(rewards))
        if mean_rew > best_mean_reward:
            best_mean_reward = mean_rew
            th.save(airl.reward_train.state_dict(), best_ckpt_dir / "reward_net.pt")
            learner.save(best_ckpt_dir / "learner_policy")
            print(f"\n[ckpt] round {round_num}: new best env reward = {mean_rew:.1f} → saved to {best_ckpt_dir}")

    airl.train(total_timesteps=total_timesteps, callback=checkpoint_callback)
    venv.seed(42)
    learner_rewards_after_training, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print(f"Mean reward before training: {np.mean(learner_rewards_before_training)}")
    print(f"Mean reward after training: {np.mean(learner_rewards_after_training)}")

    # ------------------------
    # Save AIRL results
    # ------------------------

    # 1. Reward network weights (core AIRL output)
    reward_net_path = save_dir / "reward_net.pt"
    th.save(airl.reward_train.state_dict(), reward_net_path)
    print(f"Saved reward network to {reward_net_path}")

    # 2. Trained learner policy (SB3 format)
    learner_policy_path = save_dir / "learner_policy"
    learner.save(learner_policy_path)
    print(f"Saved learner policy to {learner_policy_path}.zip")

    # Save final policy also as "learner_policy.zip" in the common format for downstream use 
    final_policy_path = "/home/nicomiguel/imitation/src/imitation/scripts/SSRR/tests/airl_outputs/final_policy"
    learner.save(final_policy_path + "/learner_policy.zip")
    print(f"Saved learner policy to {final_policy_path + '/learner_policy.zip'}")

    # 3. Reward statistics
    results_path = save_dir / "results.npy"
    np.save(
        results_path,
        {
            "mean_reward_before": float(np.mean(learner_rewards_before_training)),
            "std_reward_before": float(np.std(learner_rewards_before_training)),
            "mean_reward_after": float(np.mean(learner_rewards_after_training)),
            "std_reward_after": float(np.std(learner_rewards_after_training)),
            "rewards_before": learner_rewards_before_training,
            "rewards_after": learner_rewards_after_training,
            "archive_name": _auto_archive_name,
            "total_timesteps": total_timesteps,
        },
    )
    print(f"Saved results to {results_path}")

    # 4. Experiment config
    config = {
        "archive_name": _auto_archive_name,
        "ref_mode": ref_mode,
        "ref_amplitude": ref_amplitude,
        "ref_frequency": ref_frequency,
        "ref_phase": ref_phase,
        "max_episode_seconds": max_episode_seconds,
        "dt": dt,
        "total_timesteps": total_timesteps,
        "demo_batch_size": demo_batch_size,
        "n_disc_updates_per_round": n_disc_updates_per_round,
        "disc_lr": disc_opt_kwargs["lr"],
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "target_kl": target_kl,
    }
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
