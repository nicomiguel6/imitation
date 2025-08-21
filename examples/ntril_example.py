"""Simple example demonstrating NTRIL usage."""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import visualize_noise_levels, analyze_ranking_quality
from imitation.util.logger import configure


def main():
    """Run a simple NTRIL example on CartPole."""
    print("Running NTRIL example on CartPole-v1...")
    
    # Setup environment
    def make_env():
        return gym.make("CartPole-v1")
    
    venv = DummyVecEnv([make_env])
    
    # Train expert policy (or load pre-trained)
    print("Training expert policy...")
    expert_policy = PPO("MlpPolicy", venv, verbose=0)
    expert_policy.learn(total_timesteps=10000)
    
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    expert_trajectories = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=10),
    )
    
    print(f"Generated {len(expert_trajectories)} expert trajectories")
    
    # Setup NTRIL trainer
    print("Setting up NTRIL trainer...")
    ntril_trainer = NTRILTrainer(
        demonstrations=expert_trajectories,
        venv=venv,
        noise_levels=[0.0, 0.1, 0.2, 0.3],
        n_rollouts_per_noise=5,
        mpc_horizon=5,
        disturbance_bound=0.05,
        custom_logger=configure(format_strs=["stdout"]),
    )
    
    # Train NTRIL
    print("Training NTRIL...")
    training_stats = ntril_trainer.train(
        total_timesteps=5000,
        bc_train_kwargs={"n_epochs": 5},
        irl_train_kwargs={"n_epochs": 20, "eval_interval": 5},
    )
    
    print("Training completed!")
    
    # Analyze results
    print("\nTraining Statistics:")
    for key, value in training_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Visualize noise level effects
    if hasattr(ntril_trainer, 'noisy_rollouts') and ntril_trainer.noisy_rollouts:
        print("\nGenerating visualization...")
        try:
            visualize_noise_levels(
                ntril_trainer.noisy_rollouts,
                ntril_trainer.noise_levels,
                save_path="ntril_noise_analysis.png"
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Analyze ranking quality
    if hasattr(ntril_trainer, 'ranked_dataset') and ntril_trainer.ranked_dataset:
        print("\nAnalyzing ranking quality...")
        ranking_analysis = analyze_ranking_quality(
            ntril_trainer.ranked_dataset,
            ntril_trainer.noise_levels,
        )
        
        print("Ranking Analysis:")
        for key, value in ranking_analysis.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Evaluate final policy
    print("\nEvaluating final policy...")
    try:
        final_policy = ntril_trainer.policy
        eval_trajectories = rollout.rollout(
            final_policy,
            venv,
            rollout.make_sample_until(min_episodes=5),
        )
        
        eval_returns = [sum(traj.rews) for traj in eval_trajectories]
        print(f"Final policy evaluation:")
        print(f"  Mean return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
        print(f"  Episode lengths: {[len(traj) for traj in eval_trajectories]}")
        
    except Exception as e:
        print(f"Final evaluation failed: {e}")
    
    print("\nNTRIL example completed!")


if __name__ == "__main__":
    main()
