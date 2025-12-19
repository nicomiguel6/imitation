"""
Complete NTRIL Example for MountainCarContinuous-v0

This script demonstrates using the existing NTRILTrainer class from ntril.py
to run the complete NTRIL pipeline on MountainCarContinuous-v0.

Based on the existing codebase structure in:
- imitation/scripts/NTRIL/ntril.py (NTRILTrainer)
- imitation/scripts/NTRIL/demonstration_ranked_irl.py
- imitation/scripts/NTRIL/noise_injection.py

Author: Nicolas Miguel
Date: December 2025
"""

import os
import pathlib
import functools
from typing import Optional

import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO

from imitation.data import rollout, serialize
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util, logger as imit_logger
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper


def generate_expert_demonstrations(
    env_id: str = "MountainCarContinuous-v0",
    n_episodes: int = 20,
    train_timesteps: int = 100000,
    save_dir: str = "./ntril_outputs",
    force_retrain: bool = False,
):
    """Generate expert demonstrations using PPO.

    Args:
        env_id: Gymnasium environment ID
        n_episodes: Number of expert episodes to generate
        train_timesteps: Total timesteps for training expert
        save_dir: Directory to save outputs
        force_retrain: If True, retrain even if model exists

    Returns:
        List of expert trajectories
    """
    print("\n" + "=" * 70)
    print("STEP 1: Generating Expert Demonstrations")
    print("=" * 70)

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / "expert_policy.zip"
    traj_path = save_path / "expert_trajectories"

    rng = np.random.default_rng(42)

    # Create environment
    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Train or load expert
    if not model_path.exists() or force_retrain:
        print(f"Training expert policy for {train_timesteps} timesteps...")
        expert_policy = PPO("MlpPolicy", venv, verbose=1)
        expert_policy.learn(total_timesteps=train_timesteps)
        expert_policy.save(str(model_path))
        print(f"✓ Expert policy saved to {model_path}")
    else:
        print(f"Loading existing expert policy from {model_path}")
        expert_policy = PPO.load(str(model_path), env=venv)

    # Generate demonstrations
    print(f"Generating {n_episodes} expert trajectories...")
    expert_trajectories = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng,
    )

    # Save trajectories
    serialize.save(str(traj_path), expert_trajectories)
    print(f"Generated {len(expert_trajectories)} expert trajectories")
    print(f"Saved to {traj_path}")

    venv.close()
    return expert_trajectories


def run_ntril_training(
    demonstrations,
    env_id: str = "MountainCarContinuous-v0",
    save_dir: str = "./ntril_outputs",
    noise_levels: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    n_rollouts_per_noise: int = 10,
    bc_epochs: int = 50,
    rl_total_timesteps: int = 100000,
):
    """Run NTRIL training using the NTRILTrainer class.

    Args:
        demonstrations: Expert demonstration trajectories
        env_id: Gymnasium environment ID
        save_dir: Directory to save outputs
        noise_levels: Sequence of noise levels for data augmentation
        n_rollouts_per_noise: Number of rollouts per noise level
        bc_epochs: Number of epochs for BC training
        rl_total_timesteps: Total timesteps for final RL training

    Returns:
        Trained NTRILTrainer instance
    """
    print("\n" + "=" * 70)
    print("STEP 2: Running NTRIL Training Pipeline")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Create environment
    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Setup logger
    custom_logger = imit_logger.configure(
        folder=os.path.join(save_dir, "logs"),
        format_strs=["stdout", "tensorboard", "csv"],
    )

    # Create reward network
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
    )

    # Initialize NTRIL trainer
    print("\nInitializing NTRIL Trainer...")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Rollouts per noise level: {n_rollouts_per_noise}")

    ntril_trainer = NTRILTrainer(
        demonstrations=demonstrations,
        venv=venv,
        noise_levels=noise_levels,
        n_rollouts_per_noise=n_rollouts_per_noise,
        bc_batch_size=32,
        reward_net=reward_net,
        irl_batch_size=32,
        irl_lr=1e-3,
        custom_logger=custom_logger,
        save_dir=save_dir,
    )

    # Run training
    print("\nStarting NTRIL training pipeline...")
    training_stats = ntril_trainer.train(
        total_timesteps=rl_total_timesteps,
        bc_train_kwargs={"n_epochs": bc_epochs, "progress_bar": True},
        irl_train_kwargs={},
        rl_train_kwargs={},
    )

    print("\n" + "=" * 70)
    print("NTRIL Training Complete!")
    print("=" * 70)
    print("\nTraining Statistics:")
    for stage, stats in training_stats.items():
        print(f"\n{stage.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    venv.close()
    return ntril_trainer


def evaluate_policy(
    policy,
    env_id: str = "MountainCarContinuous-v0",
    n_episodes: int = 10,
):
    """Evaluate a trained policy.

    Args:
        policy: Policy to evaluate
        env_id: Gymnasium environment ID
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating Trained Policy")
    print("=" * 70)

    rng = np.random.default_rng(42)

    venv = util.make_vec_env(env_id, rng=rng, n_envs=1)

    eval_trajectories = rollout.rollout(
        policy,
        venv,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng,
    )

    returns = [sum(traj.rews) for traj in eval_trajectories]
    lengths = [len(traj) for traj in eval_trajectories]

    metrics = {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")

    venv.close()
    return metrics


def main():
    """Run the complete NTRIL example."""
    # Configuration
    env_id = "MountainCarContinuous-v0"
    save_dir = "./ntril_mountain_car_outputs"
    device = "mps"

    print("\n" + "=" * 70)
    print("NTRIL PIPELINE FOR MOUNTAINCARCONTINUOUS-V0")
    print("=" * 70)
    print(f"\nEnvironment: {env_id}")
    print(f"Save directory: {save_dir}")

    # Step 1: Generate expert demonstrations
    demonstrations = generate_expert_demonstrations(
        env_id=env_id,
        n_episodes=20,
        train_timesteps=100000,
        save_dir=save_dir,
        force_retrain=False,
    )

    # Step 2: Run NTRIL training
    ntril_trainer = run_ntril_training(
        demonstrations=demonstrations,
        env_id=env_id,
        save_dir=save_dir,
        noise_levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        n_rollouts_per_noise=10,
        bc_epochs=50,
        rl_total_timesteps=100000,
    )

    # Step 3: Evaluate the trained policy
    eval_metrics = evaluate_policy(
        policy=ntril_trainer.policy,
        env_id=env_id,
        n_episodes=10,
    )

    print("\n" + "=" * 70)
    print("NTRIL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {save_dir}")
    print("\nKey files:")
    print(f"  - Expert policy: {save_dir}/expert_policy.zip")
    print(f"  - Expert trajectories: {save_dir}/expert_trajectories/")
    print(f"  - BC policy: {save_dir}/BC_policy/")
    print(f"  - Training logs: {save_dir}/logs/")


if __name__ == "__main__":
    main()
