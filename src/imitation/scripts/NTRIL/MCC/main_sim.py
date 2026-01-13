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
from pathlib import Path
import functools
from typing import Optional

import numpy as np
import torch as th
import gymnasium as gym
import dataclasses
from stable_baselines3 import PPO

from imitation.data import rollout, serialize
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util, logger as imit_logger
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper


def generate_expert_demonstrations(
    env_id: str = "MountainCarContinuous-v0",
    device: str = "cuda",
    n_episodes: int = 20,
    train_timesteps: int = 10_000_000,
    checkpoint_interval: int = 1_000_000,
    use_checkpoint_at: Optional[int] = None,
    force_retrain: bool = False,
):
    """Generate demonstrations using PPO trained to varying degrees of optimality."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating Expert Demonstrations")
    print("=" * 70)

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    # Define all paths relative to script directory
    DEBUG_DIR = SCRIPT_DIR / "debug"
    CHECKPOINTS_DIR = DEBUG_DIR / "policy_checkpoints"
    DEMOS_DIR = DEBUG_DIR / "demonstrations"
    
    # Create directories
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Create environment
    venv = util.make_vec_env(
        env_id,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # Determine which checkpoint to use
    if use_checkpoint_at is None:
        checkpoint_timesteps = train_timesteps
        model_name = f"expert_policy_final_{train_timesteps}"
        MODEL_PATH = CHECKPOINTS_DIR / f"{model_name}.zip"
    else:
        checkpoint_timesteps = use_checkpoint_at
        model_name = f"expert_policy_checkpoint_{use_checkpoint_at}"
        # CheckpointCallback saves as: prefix_<steps>_steps.zip
        MODEL_PATH = CHECKPOINTS_DIR / f"expert_policy_checkpoint_{use_checkpoint_at}_steps.zip"
    
    DEMO_PATH = DEMOS_DIR / f"{model_name}_demos.pkl"

    print(f"\nConfiguration:")
    print(f"  Training timesteps: {train_timesteps}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print(f"  Using checkpoint at: {checkpoint_timesteps} steps")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Demos path: {DEMO_PATH}")

    # ============================================================
    # CHECK 1: Skip everything if demos already exist
    # ============================================================
    if DEMO_PATH.exists() and not force_retrain:
        print(f"\n✓ Found existing demonstrations at {DEMO_PATH}")
        print(f"  Loading {DEMO_PATH}...")
        expert_trajectories = serialize.load(str(DEMO_PATH))
        print(f"✓ Loaded {len(expert_trajectories)} demonstration trajectories")
        
        # Display stats
        returns = [sum(traj.rews) for traj in expert_trajectories]
        lengths = [len(traj) for traj in expert_trajectories]
        print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        
        venv.close()
        return expert_trajectories

    # ============================================================
    # CHECK 2: Skip training if specific checkpoint exists
    # ============================================================
    if MODEL_PATH.exists() and not force_retrain:
        print(f"\n✓ Found existing checkpoint at {MODEL_PATH}")
        print(f"  Loading policy from checkpoint...")
        expert_policy = PPO.load(str(MODEL_PATH), env=venv, device=device)
        print(f"✓ Loaded policy checkpoint (trained for {checkpoint_timesteps} steps)")
    else:
        # ============================================================
        # TRAIN: No checkpoint exists or force_retrain=True
        # ============================================================
        print(f"\n⚠ Checkpoint not found at {MODEL_PATH}")
        print(f"  Training expert policy with checkpoints every {checkpoint_interval} steps...")
        
        from stable_baselines3.common.callbacks import CheckpointCallback
        
        # Callback to save model at intervals
        checkpoint_callback = CheckpointCallback(
            save_freq=max(checkpoint_interval // venv.num_envs, 1),
            save_path=str(CHECKPOINTS_DIR),
            name_prefix="expert_policy_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        
        # Train the policy
        expert_policy = PPO("MlpPolicy", venv, verbose=1, device=device)
        expert_policy.learn(
            total_timesteps=train_timesteps,
            callback=checkpoint_callback,
        )
        
        # Save final model
        final_path = CHECKPOINTS_DIR / f"expert_policy_final_{train_timesteps}.zip"
        expert_policy.save(str(final_path))
        print(f"\n✓ Training complete!")
        print(f"✓ Final policy saved to {final_path}")
        print(f"✓ Checkpoints saved in {CHECKPOINTS_DIR}")
        
        # List all saved checkpoints
        checkpoints = sorted(CHECKPOINTS_DIR.glob("expert_policy_checkpoint_*_steps.zip"))
        print(f"\nAvailable checkpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            # Extract step number: "expert_policy_checkpoint_10000_steps.zip" -> "10000"
            parts = ckpt.stem.split("_")
            if len(parts) >= 5:
                step_str = parts[-2]  # Second-to-last part is the step number
                print(f"  - {step_str} steps: {ckpt.name}")
        
        # ============================================================
        # LOAD: After training, load the requested checkpoint
        # ============================================================
        if use_checkpoint_at is not None and use_checkpoint_at < train_timesteps:
            print(f"\n⚠ Requested checkpoint at {use_checkpoint_at} steps")
            if MODEL_PATH.exists():
                print(f"  Loading {MODEL_PATH}...")
                expert_policy = PPO.load(str(MODEL_PATH), env=venv, device=device)
                print(f"✓ Loaded checkpoint")
            else:
                print(f"  ERROR: Checkpoint not found after training!")
                print(f"  Using final trained policy instead.")

    # ============================================================
    # GENERATE DEMONSTRATIONS
    # ============================================================
    print(f"\nGenerating {n_episodes} demonstration trajectories...")
    expert_trajectories = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng,
    )

    # Calculate and display trajectory statistics
    returns = [sum(traj.rews) for traj in expert_trajectories]
    lengths = [len(traj) for traj in expert_trajectories]
    
    print(f"\n✓ Generated {len(expert_trajectories)} demonstration trajectories")
    print(f"  Policy trained for: {checkpoint_timesteps} steps")
    print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

    # Save trajectories
    serialize.save(str(DEMO_PATH), expert_trajectories)
    print(f"✓ Saved demonstrations to {DEMO_PATH}")

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
        custom_logger=custom_logger,
        noise_levels=noise_levels,
        n_rollouts_per_noise=n_rollouts_per_noise,
        bc_batch_size=32,
        reward_net=reward_net,
        irl_batch_size=32,
        irl_lr=1e-3,
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
    print("Generating expert demonstrations on seals' MountainCar-v0...")

    # Get the directory where THIS script is located
    SCRIPT_DIR = Path(__file__).parent.resolve()

    SAVE_DIR = SCRIPT_DIR / "ntril_outputs"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    if device == "mps":
        th.set_default_dtype(th.float32)
    else:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    rngs = np.random.default_rng()
    env_id = "seals/MountainCar-v0"

    print("\n" + "=" * 70)
    print("NTRIL PIPELINE FOR MOUNTAINCARCONTINUOUS-V0")
    print("=" * 70)
    print(f"\nEnvironment: {env_id}")

    # Step 1: Generate demonstrations (suboptimal) 
    desired_steps = 1_000_000
    demonstrations = generate_expert_demonstrations(
            env_id=env_id,
            device=str(device),
            n_episodes=20,
            train_timesteps=1_100_000,      # Train fully
            checkpoint_interval=1_000_000,   # Save every 10k steps
            use_checkpoint_at=desired_steps,     # Use 30% trained policy (SUBOPTIMAL!)
            force_retrain=False,
        )

    # Step 1.5: Collect reward statistics on suboptimal demonstrations
    print("\nCollecting reward statistics on suboptimal demonstrations...")
    returns = [sum(traj.rews) for traj in demonstrations]
    lengths = [len(traj) for traj in demonstrations]
    print(f"  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")



    # Step 2: Run NTRIL training on suboptimal demos
    ntril_trainer = run_ntril_training(
        demonstrations=demonstrations,
        env_id=env_id,
        save_dir=str(SAVE_DIR),
        noise_levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        n_rollouts_per_noise=10,
        bc_epochs=50,
        rl_total_timesteps=100_000,
    )

    # # Step 3: Evaluate the trained policy
    # eval_metrics = evaluate_policy(
    #     policy=ntril_trainer.policy,
    #     env_id=env_id,
    #     n_episodes=10,
    # )

    print("\n" + "=" * 70)
    print("NTRIL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {SAVE_DIR}")
    print("\nKey files:")
    print(f"  - Policy checkpoints: {SCRIPT_DIR / 'debug' / 'policy_checkpoints'}")
    print(f"  - Suboptimal demonstrations: {SCRIPT_DIR / 'debug' / 'demonstrations'}")
    print(f"  - Training logs: {SAVE_DIR / 'logs'}")


if __name__ == "__main__":
    main()
