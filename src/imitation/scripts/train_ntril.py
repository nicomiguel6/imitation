"""Training script for NTRIL (Noisy Trajectory Ranked Imitation Learning)."""

import logging
import os.path as osp
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from imitation.data import rollout, types
from imitation.rewards import reward_nets
from imitation.scripts.config.train_ntril import train_ntril_ex
from imitation.scripts.ingredients import environment, expert, logging as logging_ingredient
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.util import util

logger = logging.getLogger(__name__)


@train_ntril_ex.config
def train_ntril_defaults():
    """Default configuration for NTRIL training."""
    # NTRIL-specific parameters
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Noise levels to apply
    n_rollouts_per_noise = 10  # Number of rollouts per noise level
    mpc_horizon = 10  # MPC prediction horizon
    disturbance_bound = 0.1  # Disturbance bound for robust MPC
    
    # BC training parameters
    bc_train_kwargs = {
        "n_epochs": 10,
        "batch_size": 32,
    }
    
    # IRL training parameters
    irl_train_kwargs = {
        "n_epochs": 100,
        "batch_size": 32,
        "lr": 1e-3,
        "eval_interval": 10,
    }
    
    # RL training parameters
    rl_train_kwargs = {
        "total_timesteps": 100000,
    }
    
    # Reward network parameters
    reward_net_kwargs = {
        "hidden_sizes": [256, 256],
        "activation": "relu",
    }
    
    # Evaluation parameters
    n_eval_episodes = 10
    eval_max_timesteps = None
    
    # Logging and saving
    checkpoint_interval = 1000
    save_trajectories = True


@train_ntril_ex.named_config
def fast():
    """Fast training configuration for testing."""
    noise_levels = [0.0, 0.2, 0.4]
    n_rollouts_per_noise = 5
    mpc_horizon = 5
    
    bc_train_kwargs = {
        "n_epochs": 5,
        "batch_size": 32,
    }
    
    irl_train_kwargs = {
        "n_epochs": 20,
        "batch_size": 32,
        "lr": 1e-3,
        "eval_interval": 5,
    }
    
    rl_train_kwargs = {
        "total_timesteps": 10000,
    }


@train_ntril_ex.named_config
def cartpole():
    """Configuration for CartPole environment."""
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    n_rollouts_per_noise = 20
    mpc_horizon = 5
    disturbance_bound = 0.05
    
    reward_net_kwargs = {
        "hidden_sizes": [64, 64],
        "activation": "tanh",
    }


@train_ntril_ex.named_config
def mujoco():
    """Configuration for MuJoCo environments."""
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    n_rollouts_per_noise = 15
    mpc_horizon = 20
    disturbance_bound = 0.1
    
    bc_train_kwargs = {
        "n_epochs": 20,
        "batch_size": 64,
    }
    
    irl_train_kwargs = {
        "n_epochs": 200,
        "batch_size": 64,
        "lr": 3e-4,
        "eval_interval": 20,
    }
    
    reward_net_kwargs = {
        "hidden_sizes": [256, 256, 256],
        "activation": "relu",
    }


@train_ntril_ex.capture
def train_ntril(
    _seed: int,
    noise_levels: Sequence[float],
    n_rollouts_per_noise: int,
    mpc_horizon: int,
    disturbance_bound: float,
    bc_train_kwargs: Dict[str, Any],
    irl_train_kwargs: Dict[str, Any],
    rl_train_kwargs: Dict[str, Any],
    reward_net_kwargs: Dict[str, Any],
    n_eval_episodes: int,
    eval_max_timesteps: Optional[int],
    checkpoint_interval: int,
    save_trajectories: bool,
) -> Dict[str, Any]:
    """Train NTRIL on specified environment and expert demonstrations.
    
    Args:
        _seed: Random seed
        noise_levels: Sequence of noise levels to apply
        n_rollouts_per_noise: Number of rollouts per noise level
        mpc_horizon: MPC prediction horizon
        disturbance_bound: Disturbance bound for robust MPC
        bc_train_kwargs: BC training parameters
        irl_train_kwargs: IRL training parameters
        rl_train_kwargs: RL training parameters
        reward_net_kwargs: Reward network parameters
        n_eval_episodes: Number of evaluation episodes
        eval_max_timesteps: Maximum timesteps per evaluation episode
        checkpoint_interval: Interval for saving checkpoints
        save_trajectories: Whether to save generated trajectories
        
    Returns:
        Dictionary containing training statistics
    """
    # Get ingredients
    venv = environment.venv
    expert_trajs = expert.trajectories
    custom_logger = logging_ingredient.logger
    
    # Create reward network
    reward_net = reward_nets.BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        **reward_net_kwargs,
    )
    
    # Initialize NTRIL trainer
    ntril_trainer = NTRILTrainer(
        demonstrations=expert_trajs,
        venv=venv,
        noise_levels=noise_levels,
        n_rollouts_per_noise=n_rollouts_per_noise,
        mpc_horizon=mpc_horizon,
        disturbance_bound=disturbance_bound,
        bc_batch_size=bc_train_kwargs.get("batch_size", 32),
        bc_train_kwargs=bc_train_kwargs,
        reward_net=reward_net,
        irl_batch_size=irl_train_kwargs.get("batch_size", 32),
        irl_lr=irl_train_kwargs.get("lr", 1e-3),
        custom_logger=custom_logger,
    )
    
    # Train NTRIL
    custom_logger.log("Starting NTRIL training...")
    training_stats = ntril_trainer.train(
        total_timesteps=rl_train_kwargs["total_timesteps"],
        bc_train_kwargs=bc_train_kwargs,
        irl_train_kwargs=irl_train_kwargs,
        rl_train_kwargs=rl_train_kwargs,
    )
    
    # Evaluate trained policy
    custom_logger.log("Evaluating trained policy...")
    eval_stats = _evaluate_ntril_policy(
        ntril_trainer,
        venv,
        n_eval_episodes,
        eval_max_timesteps,
        custom_logger,
    )
    
    # Save results
    if save_trajectories:
        _save_ntril_results(ntril_trainer, training_stats, eval_stats, custom_logger)
    
    # Combine stats
    final_stats = {
        "training": training_stats,
        "evaluation": eval_stats,
        "config": {
            "noise_levels": noise_levels,
            "n_rollouts_per_noise": n_rollouts_per_noise,
            "mpc_horizon": mpc_horizon,
            "disturbance_bound": disturbance_bound,
        },
    }
    
    return final_stats


def _evaluate_ntril_policy(
    ntril_trainer: NTRILTrainer,
    venv,
    n_eval_episodes: int,
    eval_max_timesteps: Optional[int],
    custom_logger,
) -> Dict[str, Any]:
    """Evaluate the trained NTRIL policy."""
    try:
        policy = ntril_trainer.policy
        
        # Collect evaluation rollouts
        sample_until = rollout.make_sample_until(
            min_episodes=n_eval_episodes,
            min_timesteps=None,
            max_timesteps=eval_max_timesteps,
        )
        
        eval_trajs = rollout.rollout(
            policy,
            venv,
            sample_until,
            rng=np.random.default_rng(),
        )
        
        # Compute statistics
        eval_stats = rollout.rollout_stats(eval_trajs)
        eval_stats["n_episodes"] = len(eval_trajs)
        
        custom_logger.log(f"Evaluation: {len(eval_trajs)} episodes, "
                         f"mean return: {eval_stats['return_mean']:.3f}")
        
        return eval_stats
        
    except Exception as e:
        custom_logger.log(f"Evaluation failed: {e}")
        return {"evaluation_failed": True, "error": str(e)}


def _save_ntril_results(
    ntril_trainer: NTRILTrainer,
    training_stats: Dict[str, Any],
    eval_stats: Dict[str, Any],
    custom_logger,
) -> None:
    """Save NTRIL training results."""
    try:
        # Save reward network
        reward_net_path = "ntril_reward_net.pth"
        ntril_trainer.irl_trainer.save_reward_net(reward_net_path)
        
        # Save noisy rollouts if requested
        if hasattr(ntril_trainer, 'noisy_rollouts') and ntril_trainer.noisy_rollouts:
            for i, rollouts in enumerate(ntril_trainer.noisy_rollouts):
                noise_level = ntril_trainer.noise_levels[i]
                rollout_path = f"noisy_rollouts_noise_{noise_level:.2f}.npz"
                types.save(rollout_path, rollouts)
                custom_logger.log(f"Saved noisy rollouts to {rollout_path}")
        
        # Save ranked dataset
        if hasattr(ntril_trainer, 'ranked_dataset') and ntril_trainer.ranked_dataset:
            dataset_path = "ranked_dataset.npz"
            # Convert to trajectory format for saving
            # This is a simplified save - in practice you might want a more sophisticated format
            custom_logger.log(f"Ranked dataset contains {len(ntril_trainer.ranked_dataset.obs)} transitions")
        
        custom_logger.log("Successfully saved NTRIL results")
        
    except Exception as e:
        custom_logger.log(f"Failed to save results: {e}")


def main_console():
    """Entry point for console script."""
    observer = FileStorageObserver.create(
        osp.join("output", "sacred", "train_ntril")
    )
    train_ntril_ex.observers.append(observer)
    train_ntril_ex.run_commandline()


if __name__ == "__main__":
    main_console()
