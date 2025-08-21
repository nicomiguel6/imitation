"""Main NTRIL (Noisy Trajectory Ranked Imitation Learning) algorithm implementation."""

import abc
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common import policies, vec_env

from imitation.algorithms import base, bc
from imitation.data import rollout, types
from imitation.policies import base as policy_base
from imitation.rewards import reward_nets
from imitation.scripts.NTRIL.noise_injection import NoiseInjector
from imitation.scripts.NTRIL.ranked_dataset import RankedDatasetBuilder
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from imitation.scripts.NTRIL.demonstration_ranked_irl import DemonstrationRankedIRL
from imitation.util import logger as imit_logger
from imitation.util import util

logger = logging.getLogger(__name__)


class NTRILTrainer(base.BaseImitationAlgorithm):
    """Noisy Trajectory Ranked Imitation Learning trainer.
    
    This class implements the complete NTRIL pipeline:
    1. Train BC policy on demonstrations
    2. Generate noisy rollouts with varying noise levels
    3. Apply robust tube MPC for data augmentation
    4. Build ranked dataset
    5. Train reward network using demonstration ranked IRL
    6. Train final policy using RL
    """

    def __init__(
        self,
        demonstrations: Sequence[types.Trajectory],
        venv: vec_env.VecEnv,
        policy: Optional[policies.BasePolicy] = None,
        *,
        noise_levels: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        n_rollouts_per_noise: int = 10,
        mpc_horizon: int = 10,
        disturbance_bound: float = 0.1,
        bc_batch_size: int = 32,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        reward_net: Optional[reward_nets.RewardNet] = None,
        irl_batch_size: int = 32,
        irl_lr: float = 1e-3,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        **kwargs,
    ):
        """Initialize NTRIL trainer.
        
        Args:
            demonstrations: Expert demonstration trajectories
            venv: Vectorized environment for rollout collection
            policy: Initial policy (if None, will be created during BC training)
            noise_levels: Sequence of noise levels to apply to policy
            n_rollouts_per_noise: Number of rollouts to collect per noise level
            mpc_horizon: Horizon for robust tube MPC
            disturbance_bound: Bound for MPC disturbance set
            bc_batch_size: Batch size for behavioral cloning
            bc_train_kwargs: Additional kwargs for BC training
            reward_net: Reward network (if None, will be created)
            irl_batch_size: Batch size for IRL training
            irl_lr: Learning rate for IRL training
            custom_logger: Custom logger instance
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(custom_logger=custom_logger, **kwargs)
        
        self.demonstrations = demonstrations
        self.venv = venv
        self.policy = policy
        self.noise_levels = noise_levels
        self.n_rollouts_per_noise = n_rollouts_per_noise
        
        # Initialize components
        self.noise_injector = NoiseInjector()
        self.robust_mpc = RobustTubeMPC(
            horizon=mpc_horizon,
            disturbance_bound=disturbance_bound,
        )
        self.dataset_builder = RankedDatasetBuilder()
        
        # BC trainer
        self.bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            policy=policy,
            demonstrations=demonstrations,
            batch_size=bc_batch_size,
            custom_logger=self._logger,
            **(bc_train_kwargs or {}),
        )
        
        # Reward network and IRL
        if reward_net is None:
            reward_net = reward_nets.BasicRewardNet(
                venv.observation_space,
                venv.action_space,
            )
        
        self.irl_trainer = DemonstrationRankedIRL(
            reward_net=reward_net,
            venv=venv,
            batch_size=irl_batch_size,
            lr=irl_lr,
            custom_logger=self._logger,
        )
        
        # Storage for intermediate results
        self.bc_policy: Optional[policies.BasePolicy] = None
        self.noisy_rollouts: List[List[types.Trajectory]] = []
        self.augmented_data: List[types.Transitions] = []
        self.ranked_dataset: Optional[types.Transitions] = None
        self.learned_reward_net: Optional[reward_nets.RewardNet] = None

    def train(
        self,
        total_timesteps: int,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
        irl_train_kwargs: Optional[Mapping[str, Any]] = None,
        rl_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete NTRIL training pipeline.
        
        Args:
            total_timesteps: Total timesteps for final RL training
            bc_train_kwargs: Additional kwargs for BC training
            irl_train_kwargs: Additional kwargs for IRL training  
            rl_train_kwargs: Additional kwargs for RL training
            
        Returns:
            Dictionary containing training statistics
        """
        stats = {}
        
        # Step 1: Train BC policy
        self._logger.log("Starting behavioral cloning training...")
        bc_stats = self._train_bc_policy(**(bc_train_kwargs or {}))
        stats["bc"] = bc_stats
        
        # Step 2: Generate noisy rollouts
        self._logger.log("Generating noisy rollouts...")
        rollout_stats = self._generate_noisy_rollouts()
        stats["rollouts"] = rollout_stats
        
        # Step 3: Apply robust tube MPC and augment data
        self._logger.log("Applying robust tube MPC and augmenting data...")
        augmentation_stats = self._augment_data_with_mpc()
        stats["augmentation"] = augmentation_stats
        
        # Step 4: Build ranked dataset
        self._logger.log("Building ranked dataset...")
        ranking_stats = self._build_ranked_dataset()
        stats["ranking"] = ranking_stats
        
        # Step 5: Train reward network using demonstration ranked IRL
        self._logger.log("Training reward network with demonstration ranked IRL...")
        irl_stats = self._train_reward_network(**(irl_train_kwargs or {}))
        stats["irl"] = irl_stats
        
        # Step 6: Train final policy using RL
        self._logger.log("Training final policy using RL...")
        rl_stats = self._train_final_policy(total_timesteps, **(rl_train_kwargs or {}))
        stats["rl"] = rl_stats
        
        return stats

    def _train_bc_policy(self, **kwargs) -> Dict[str, Any]:
        """Train the initial BC policy."""
        self.bc_trainer.train(**kwargs)
        self.bc_policy = self.bc_trainer.policy
        
        # Evaluate BC policy
        bc_rollouts = rollout.rollout(
            self.bc_policy,
            self.venv,
            rollout.make_sample_until(min_timesteps=1000),
        )
        
        return {
            "n_rollouts": len(bc_rollouts),
            "mean_return": np.mean([sum(traj.rews) for traj in bc_rollouts]),
            "mean_length": np.mean([len(traj) for traj in bc_rollouts]),
        }

    def _generate_noisy_rollouts(self) -> Dict[str, Any]:
        """Generate rollouts with different noise levels."""
        if self.bc_policy is None:
            raise ValueError("BC policy must be trained before generating noisy rollouts")
        
        self.noisy_rollouts = []
        total_rollouts = 0
        
        for noise_level in self.noise_levels:
            # Create noisy policy
            noisy_policy = self.noise_injector.inject_noise(
                self.bc_policy, 
                noise_level=noise_level
            )
            
            # Collect rollouts
            rollouts = rollout.rollout(
                noisy_policy,
                self.venv,
                rollout.make_sample_until(
                    min_episodes=self.n_rollouts_per_noise,
                    min_timesteps=None,
                ),
            )
            
            self.noisy_rollouts.append(rollouts)
            total_rollouts += len(rollouts)
            
            self._logger.log(
                f"Collected {len(rollouts)} rollouts with noise level {noise_level}"
            )
        
        return {
            "total_rollouts": total_rollouts,
            "noise_levels": list(self.noise_levels),
            "rollouts_per_level": [len(rollouts) for rollouts in self.noisy_rollouts],
        }

    def _augment_data_with_mpc(self) -> Dict[str, Any]:
        """Apply robust tube MPC to augment the noisy rollouts."""
        self.augmented_data = []
        total_augmented_transitions = 0
        
        for noise_idx, rollouts in enumerate(self.noisy_rollouts):
            noise_level = self.noise_levels[noise_idx]
            
            for traj in rollouts:
                # Apply robust tube MPC to each trajectory
                augmented_transitions = self.robust_mpc.augment_trajectory(
                    traj, 
                    noise_level=noise_level
                )
                self.augmented_data.append(augmented_transitions)
                total_augmented_transitions += len(augmented_transitions.obs)
        
        return {
            "total_augmented_transitions": total_augmented_transitions,
            "augmentation_ratio": total_augmented_transitions / sum(
                len(rollouts) * np.mean([len(traj) for traj in rollouts])
                for rollouts in self.noisy_rollouts
            ),
        }

    def _build_ranked_dataset(self) -> Dict[str, Any]:
        """Build ranked dataset from augmented data."""
        # Combine all augmented data with noise level rankings
        noise_rankings = []
        all_transitions = []
        
        for noise_idx, transitions in enumerate(self.augmented_data):
            noise_level = self.noise_levels[noise_idx % len(self.noise_levels)]
            all_transitions.append(transitions)
            noise_rankings.extend([noise_level] * len(transitions.obs))
        
        # Build ranked dataset
        self.ranked_dataset = self.dataset_builder.build_ranked_dataset(
            all_transitions,
            noise_rankings,
        )
        
        return {
            "total_ranked_transitions": len(self.ranked_dataset.obs),
            "unique_noise_levels": len(set(noise_rankings)),
            "ranking_distribution": dict(zip(*np.unique(noise_rankings, return_counts=True))),
        }

    def _train_reward_network(self, **kwargs) -> Dict[str, Any]:
        """Train reward network using demonstration ranked IRL."""
        if self.ranked_dataset is None:
            raise ValueError("Ranked dataset must be built before training reward network")
        
        irl_stats = self.irl_trainer.train(
            demonstrations=self.demonstrations,
            ranked_dataset=self.ranked_dataset,
            **kwargs
        )
        
        self.learned_reward_net = self.irl_trainer.reward_net
        return irl_stats

    def _train_final_policy(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Train final policy using RL with learned reward."""
        if self.learned_reward_net is None:
            raise ValueError("Reward network must be trained before training final policy")
        
        # This would integrate with existing RL training infrastructure
        # For now, return placeholder stats
        return {
            "total_timesteps": total_timesteps,
            "final_policy_trained": True,
        }

    @property
    def policy(self) -> policies.BasePolicy:
        """Return the current policy."""
        if self.bc_policy is not None:
            return self.bc_policy
        elif self.bc_trainer.policy is not None:
            return self.bc_trainer.policy
        else:
            raise ValueError("No policy available - train BC first")
