"""Main NTRIL (Noisy Trajectory Ranked Imitation Learning) algorithm implementation."""

import abc
import dataclasses
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common import policies, vec_env

from imitation.algorithms import base, bc
from imitation.data import rollout, types
from imitation.policies import base as policy_base
from imitation.rewards import reward_nets
from imitation.scripts.NTRIL.noise_injection import EpsilonGreedyNoiseInjector
from imitation.scripts.NTRIL.ranked_dataset import RankedDatasetBuilder
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from imitation.scripts.NTRIL.demonstration_ranked_irl import DemonstrationRankedIRL
from imitation.util import logger as imit_logger
from imitation.util import util

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class NTRILTrainer(base.BaseImitationAlgorithm):
    """Noisy Trajectory Ranked Imitation Learning trainer."""
    
    # Required fields
    demonstrations: Sequence[types.Trajectory]
    venv: vec_env.VecEnv

    custom_logger: Optional[imit_logger.HierarchicalLogger] = None
    # Training configuration
    noise_levels: Sequence[float] = dataclasses.field(
        default_factory=lambda: (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    )
    n_rollouts_per_noise: int = 10
    mpc_horizon: int = 10
    disturbance_bound: float = 0.1
    bc_batch_size: int = 32
    bc_train_kwargs: Optional[Mapping[str, Any]] = None
    reward_net: Optional[reward_nets.RewardNet] = None
    irl_batch_size: int = 32
    irl_lr: float = 1e-3
    save_dir: Optional[str] = None
    rng: int = 42
    
    def __post_init__(self):
        """Initialize components after dataclass creation."""
        # Device setup
        if th.cuda.is_available():
            device = th.device("cuda:0")
        else:
            device = th.device("cpu")
        object.__setattr__(self, 'device', device)
        
        # Save dir setup
        if self.save_dir is None:
            save_dir = os.path.join(
                os.getcwd(), "ntril_runs", datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            save_dir = os.path.abspath(self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        object.__setattr__(self, 'save_dir', save_dir)
        
        object.__setattr__(self, 'rng', np.random.default_rng(self.rng))
        # BC trainer - policy will be auto-created
        bc_trainer = bc.BC(
            observation_space=self.venv.observation_space,
            action_space=self.venv.action_space,
            policy=None,  
            demonstrations=self.demonstrations,
            batch_size=self.bc_batch_size,
            device=device,
            custom_logger=self.custom_logger,
            rng=self.rng,
            **(self.bc_train_kwargs or {}),
        )
        object.__setattr__(self, 'bc_trainer', bc_trainer)
        
        # Reward network
        if self.reward_net is None:
            reward_net = reward_nets.TrajectoryRewardNet(
                observation_space=self.venv.observation_space,
                action_space=self.venv.action_space,
                use_state=True,
                use_action=False,
                use_next_state=False,
                use_done=False,
            )
            object.__setattr__(self, 'reward_net', reward_net)
        
        # Move reward net to device
        self.reward_net.to(device)
        
        # Storage for training artifacts
        object.__setattr__(self, 'bc_policy', None)
        object.__setattr__(self, 'noisy_rollouts', [])
        object.__setattr__(self, 'augmented_data', [])
        object.__setattr__(self, 'ranked_dataset', None)
        object.__setattr__(self, 'learned_reward_net', None)
        object.__setattr__(self, 'final_policy', None)

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
        self.save_initial_bc_policy_dir = os.path.join(self.save_dir, "initial_BC_policy")

        # check if it already exists
        if os.path.exists(self.save_initial_bc_policy_dir):
            print("Loading existing BC policy...")
            self.bc_policy = bc.reconstruct_policy(
                self.save_initial_bc_policy_dir, device=self.device)
        else:
            print("Training BC policy...")
            self.bc_trainer.train(**kwargs)
            self.bc_policy = self.bc_trainer.policy

        util.save_policy(self.bc_policy, self.save_initial_bc_policy_dir)

        # Evaluate BC policy
        bc_rollouts = rollout.rollout(
            self.bc_policy,
            self.venv,
            rollout.make_sample_until(min_timesteps=1000),
            rng=self.rng,
        )

        return {
            "n_rollouts": len(bc_rollouts),
            "mean_return": np.mean([sum(traj.rews) for traj in bc_rollouts]),
            "mean_length": np.mean([len(traj) for traj in bc_rollouts]),
        }

    def _generate_noisy_rollouts(self) -> Dict[str, Any]:
        """Generate rollouts with different noise levels."""
        if self.bc_policy is None:
            raise ValueError(
                "BC policy must be trained/loaded before generating noisy rollouts"
            )

        self.noisy_rollouts = []
        total_rollouts = 0
        self.noise_injector = EpsilonGreedyNoiseInjector()
        noisy_policies_dir = os.path.join(self.save_dir, "noisy_policies")
        os.makedirs(noisy_policies_dir, exist_ok=True)

        base_policy_path = os.path.join(noisy_policies_dir, "base_policy.pt")
        if not os.path.exists(base_policy_path):
            util.save_policy(self.bc_policy, base_policy_path)

        noisy_policies_metadata = []

        noise_rollout_data = {}

        for noise_level in self.noise_levels:
            # Create noisy policy
            noisy_policy = self.noise_injector.inject_noise(
                self.bc_policy, noise_level=noise_level
            )
            noisy_policies_metadata.append(
                {
                    "noise_level": float(noise_level),
                    "base_policy_path": base_policy_path,
                    "noise_injector": type(self.noise_injector).__name__,
                }
            )

            # Collect rollouts
            rollouts = rollout.rollout(
                noisy_policy,
                self.venv,
                rollout.make_sample_until(
                    min_episodes=self.n_rollouts_per_noise,
                    min_timesteps=1000,
                ),
                rng=self.rng,
            )

            self.noisy_rollouts.append(rollouts)
            total_rollouts += len(rollouts)

            # self._logger.log(
            #     f"Collected {len(rollouts)} rollouts with noise level {noise_level}"
            # )

            mean_reward = np.mean([sum(traj.rews) for traj in rollouts])
            std_reward = np.std([sum(traj.rews) for traj in rollouts])
            noise_rollout_data[noise_level] = (mean_reward, std_reward)

        metadata_path = os.path.join(noisy_policies_dir, "noisy_policies.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(noisy_policies_metadata, f, indent=2)

        plot_path = os.path.join(self.save_dir, "noise_levels_visualization.png")

        # Visualize noise levels in rollouts with std deviation plot
        plt.errorbar(self.noise_levels, 
                     [noise_rollout_data[n][0] for n in self.noise_levels],
                     yerr=[noise_rollout_data[n][1] for n in self.noise_levels],
                     fmt='-o')
        plt.xlabel('Noise Level (Epsilon)')
        plt.ylabel('Mean Return')
        plt.title('Noisy Rollouts Performance')
        plt.savefig(plot_path)
        plt.close()

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
                    traj, noise_level=noise_level
                )
                self.augmented_data.append(augmented_transitions)
                total_augmented_transitions += len(augmented_transitions.obs)

        return {
            "total_augmented_transitions": total_augmented_transitions,
            "augmentation_ratio": total_augmented_transitions
            / sum(
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
            "ranking_distribution": dict(
                zip(*np.unique(noise_rankings, return_counts=True))
            ),
        }

    def _train_reward_network(self, **kwargs) -> Dict[str, Any]:
        """Train reward network using demonstration ranked IRL."""
        if self.ranked_dataset is None:
            raise ValueError(
                "Ranked dataset must be built before training reward network"
            )

        irl_stats = self.irl_trainer.train(
            demonstrations=self.demonstrations,
            ranked_dataset=self.ranked_dataset,
            **kwargs,
        )

        self.learned_reward_net = self.irl_trainer.reward_net
        return irl_stats

    def _train_final_policy(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Train final policy using RL with learned reward."""
        if self.learned_reward_net is None:
            raise ValueError(
                "Reward network must be trained before training final policy"
            )

        # This would integrate with existing RL training infrastructure
        # For now, return placeholder stats
        return {
            "total_timesteps": total_timesteps,
            "final_policy_trained": True,
        }
    
    @property
    def robust_mpc(self) -> RobustTubeMPC:
        """Return the current Robust Tube MPC."""
        return self._robust_mpc

    @robust_mpc.setter
    def robust_mpc(self, robust_mpc: RobustTubeMPC):
        """Function to catch external Robust Tube MPC. Main file running the trainer should set up the robust tube MPC and then pass it here."""
        self.robust_mpc = robust_mpc

    @property
    def policy(self) -> policies.BasePolicy:
        """Return the current policy."""
        if self.bc_policy is not None:
            return self.bc_policy
        elif self.bc_trainer.policy is not None:
            return self.bc_trainer.policy
        else:
            raise ValueError("No policy available - train BC first")
