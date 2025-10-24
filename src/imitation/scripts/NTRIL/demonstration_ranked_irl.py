"""Demonstration Ranked Inverse Reinforcement Learning for NTRIL pipeline."""

import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
import random
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common import vec_env
from torch.utils.data import DataLoader, Dataset

from imitation.algorithms import base
from imitation.data import rollout, types
from imitation.rewards import reward_nets
from imitation.util import logger as imit_logger
from imitation.util import networks, util


class RankedTransitionsDataset(Dataset):
    """PyTorch dataset for ranked transitions."""
    
    def __init__(
        self,
        demonstrations: List[Sequence[types.TrajectoryWithRew]],
        num_snippets: int = 10,
        min_segment_length: int = 50,
        max_segment_length: int = 100,
    ):
        """Initialize ranked transitions dataset.

        From the total set of demonstrations, a number of snippets will be generated 
        
        Args:
            demonstrations: Noisy rollout demonstration trajectories
            ranked_transitions: Transitions with ranking information
            segment_length: Length of trajectory segments for comparison
        """
        #self.ranked_transitions = ranked_transitions
        self.num_snippets = num_snippets
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length 
        self.demo_dict = {}
        self.demonstrations = []
        self.training_data = {'traj': [], 'label': []}

        self._append_demonstrations(demonstrations)
        self._build_dict()
        self._generate_training_samples()
        
        
        # Prepare expert segments
        # self.expert_segments = self._extract_expert_segments()
        
        # Prepare ranked segments
        # self.ranked_segments = self._extract_ranked_segments()
    
    def _append_demonstrations(self, demonstrations):
        """ Store all demonstrations """
        for demonstration_list in demonstrations:
            for demonstration in demonstration_list:
                self.demonstrations.append(demonstration)

    def _build_dict(self):
        """Build demonstration dictionary to track bins of different noise.

        Make sure all demonstrations have been processed to self.demonstrations before calling. 
         """
        for demonstration in self.demonstrations:
            noise_epsilon = demonstration.infos[0]["noise_level"]
            if noise_epsilon in self.demo_dict:
                self.demo_dict[noise_epsilon].append(demonstration)
            else:
                self.demo_dict[noise_epsilon] = []
                self.demo_dict[noise_epsilon].append(demonstration)
    
    def _generate_training_samples(self):
        """Generate training samples. 

        
        """
        step = 2

        # extract all noise levels
        noise_levels = list(self.demo_dict.keys())

        # Build all snippets of trajectories
        for itx in range(self.num_snippets):
            # Pick two noise levels at random (make sure they are different)
            ni = 0
            nj = 0
            while(ni == nj):
                ni = np.random.randint(len(noise_levels))
                nj = np.random.randint(len(noise_levels))
            
            # pick random trajectory from each bin
            ti = random.choice(self.demo_dict[noise_levels[ni]])
            tj = random.choice(self.demo_dict[noise_levels[nj]])

            # Create random snippet from each trajectory
            min_length = min(len(ti), len(tj))
            rand_length = np.random.randint(self.min_segment_length, self.max_segment_length)
            if ni < nj: # bin i has less noise so choose ti snippet to be later than tj
                tj_start = np.random.randint(min_length - rand_length + 1)
                ti_start = np.random.randint(tj_start, len(ti) - rand_length + 1)
            else: # tj has less noise so start later
                ti_start = np.random.randint(min_length - rand_length + 1)
                tj_start = np.random.randint(ti_start, len(tj) - rand_length + 1)
            
            snip_i = ti.obs[ti_start:ti_start+rand_length:step]
            snip_j = tj.obs[tj_start:tj_start+rand_length:step]

            if noise_levels[ni] < noise_levels[nj]:
                # bin i has less noise, so better
                label = 1
            else:
                # bin j has less noise
                label = 0
            
            self.training_data['traj'].append((snip_i, snip_j))
            self.training_data['label'].append(label)

        
    
    def __len__(self) -> int:
        """Return number of possible segment pairs."""
        return len(self.training_data['label'])
    
    def __getitem__(self, idx: int) -> Tuple[types.Transitions, types.Transitions, float]:
        """Get a specific trajectory snippet pairand label at index idx"""
        return self.training_data['traj'][idx], self.training_data['label'][idx]


class DemonstrationRankedIRL(base.BaseImitationAlgorithm):
    """Demonstration Ranked Inverse Reinforcement Learning.
    
    This class learns reward functions from ranked demonstrations and
    augmented trajectory data, using the ranking structure to improve
    reward learning.
    """
    
    def __init__(
        self,
        reward_net: reward_nets.RewardNet,
        venv: vec_env.VecEnv,
        *,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ranking_loss_weight: float = 1.0,
        preference_loss_weight: float = 1.0,
        regularization_weight: float = 1e-3,
        segment_length: int = 50,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        **kwargs,
    ):
        """Initialize Demonstration Ranked IRL.
        
        Args:
            reward_net: Reward network to train
            venv: Vectorized environment
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: Weight decay for regularization
            ranking_loss_weight: Weight for ranking loss
            preference_loss_weight: Weight for preference loss
            regularization_weight: Weight for regularization loss
            segment_length: Length of trajectory segments
            custom_logger: Custom logger instance
            **kwargs: Additional arguments for base class
        """
        super().__init__(custom_logger=custom_logger, **kwargs)
        
        self.reward_net = reward_net
        self.venv = venv
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.ranking_loss_weight = ranking_loss_weight
        self.preference_loss_weight = preference_loss_weight
        self.regularization_weight = regularization_weight
        self.segment_length = segment_length
        
        # Initialize optimizer
        self.optimizer = th.optim.Adam(
            self.reward_net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Storage for training data
        self.dataset: Optional[RankedTransitionsDataset] = None
        self.dataloader: Optional[DataLoader] = None
    
    def train(
        self,
        demonstrations: Sequence[types.Trajectory],
        ranked_dataset: types.Transitions,
        n_epochs: int = 100,
        eval_interval: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the reward network using ranked demonstrations.
        
        Args:
            demonstrations: Expert demonstration trajectories
            ranked_dataset: Ranked transitions dataset
            n_epochs: Number of training epochs
            eval_interval: Interval for evaluation
            **kwargs: Additional training arguments
            
        Returns:
            Training statistics
        """
        # Create dataset and dataloader
        self.dataset = RankedTransitionsDataset(
            demonstrations=demonstrations,
            ranked_transitions=ranked_dataset,
            segment_length=self.segment_length,
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        # Training loop
        stats = {
            "epoch_losses": [],
            "ranking_losses": [],
            "preference_losses": [],
            "regularization_losses": [],
        }
        
        for epoch in range(n_epochs):
            epoch_loss = self._train_epoch()
            stats["epoch_losses"].append(epoch_loss)
            
            if epoch % eval_interval == 0:
                self._logger.log(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
                
                # Evaluate reward network
                eval_stats = self._evaluate_reward_net(demonstrations, ranked_dataset)
                for key, value in eval_stats.items():
                    if key not in stats:
                        stats[key] = []
                    stats[key].append(value)
        
        return stats
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        if self.dataloader is None:
            raise ValueError("Dataloader not initialized")
        
        total_loss = 0.0
        n_batches = 0
        
        self.reward_net.train()
        
        for batch in self.dataloader:
            expert_segments, ranked_segments, preferences = batch
            
            # Forward pass
            loss = self._compute_loss(expert_segments, ranked_segments, preferences)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _compute_loss(
        self,
        expert_segments: List[types.Transitions],
        ranked_segments: List[types.Transitions],
        preferences: th.Tensor,
    ) -> th.Tensor:
        """Compute the total loss for a batch."""
        # Compute rewards for segments
        expert_rewards = []
        ranked_rewards = []
        
        for expert_seg, ranked_seg in zip(expert_segments, ranked_segments):
            # Expert segment rewards
            expert_obs = th.tensor(expert_seg.obs, dtype=th.float32)
            expert_acts = th.tensor(expert_seg.acts, dtype=th.float32)
            expert_next_obs = th.tensor(expert_seg.next_obs, dtype=th.float32)
            
            expert_reward = self.reward_net(expert_obs, expert_acts, expert_next_obs, None).sum()
            expert_rewards.append(expert_reward)
            
            # Ranked segment rewards
            ranked_obs = th.tensor(ranked_seg.obs, dtype=th.float32)
            ranked_acts = th.tensor(ranked_seg.acts, dtype=th.float32)
            ranked_next_obs = th.tensor(ranked_seg.next_obs, dtype=th.float32)
            
            ranked_reward = self.reward_net(ranked_obs, ranked_acts, ranked_next_obs, None).sum()
            ranked_rewards.append(ranked_reward)
        
        expert_rewards = th.stack(expert_rewards)
        ranked_rewards = th.stack(ranked_rewards)
        
        # Preference loss (Bradley-Terry model)
        preference_logits = expert_rewards - ranked_rewards
        preference_loss = F.binary_cross_entropy_with_logits(
            preference_logits, preferences
        )
        
        # Ranking loss (expert should have highest reward)
        ranking_loss = F.relu(ranked_rewards - expert_rewards + 1.0).mean()
        
        # Regularization loss
        reg_loss = sum(p.pow(2.0).sum() for p in self.reward_net.parameters())
        
        # Total loss
        total_loss = (
            self.preference_loss_weight * preference_loss +
            self.ranking_loss_weight * ranking_loss +
            self.regularization_weight * reg_loss
        )
        
        return total_loss
    
    def _collate_fn(self, batch) -> Tuple[List[types.Transitions], List[types.Transitions], th.Tensor]:
        """Collate function for DataLoader."""
        expert_segments = []
        ranked_segments = []
        preferences = []
        
        for expert_seg, ranked_seg, pref in batch:
            expert_segments.append(expert_seg)
            ranked_segments.append(ranked_seg)
            preferences.append(pref)
        
        preferences = th.tensor(preferences, dtype=th.float32)
        
        return expert_segments, ranked_segments, preferences
    
    def _evaluate_reward_net(
        self,
        demonstrations: Sequence[types.Trajectory],
        ranked_dataset: types.Transitions,
    ) -> Dict[str, float]:
        """Evaluate the reward network."""
        self.reward_net.eval()
        
        with th.no_grad():
            # Compute average reward for expert demonstrations
            expert_rewards = []
            for traj in demonstrations:
                obs = th.tensor(traj.obs[:-1], dtype=th.float32)
                acts = th.tensor(traj.acts, dtype=th.float32)
                next_obs = th.tensor(traj.obs[1:], dtype=th.float32)
                
                rewards = self.reward_net(obs, acts, next_obs, None)
                expert_rewards.append(rewards.mean().item())
            
            avg_expert_reward = np.mean(expert_rewards)
            
            # Compute average reward for ranked dataset (sample subset)
            n_samples = min(1000, len(ranked_dataset.obs))
            indices = np.random.choice(len(ranked_dataset.obs) - 1, n_samples, replace=False)
            
            obs = th.tensor(ranked_dataset.obs[indices], dtype=th.float32)
            acts = th.tensor(ranked_dataset.acts[indices], dtype=th.float32)
            next_obs = th.tensor(ranked_dataset.next_obs[indices], dtype=th.float32)
            
            ranked_rewards = self.reward_net(obs, acts, next_obs, None)
            avg_ranked_reward = ranked_rewards.mean().item()
            
            # Check if expert demonstrations have higher rewards
            reward_ordering_correct = avg_expert_reward > avg_ranked_reward
        
        return {
            "avg_expert_reward": avg_expert_reward,
            "avg_ranked_reward": avg_ranked_reward,
            "reward_gap": avg_expert_reward - avg_ranked_reward,
            "reward_ordering_correct": float(reward_ordering_correct),
        }
    
    def predict_reward(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        next_obs: np.ndarray,
    ) -> np.ndarray:
        """Predict rewards for given transitions."""
        self.reward_net.eval()
        
        with th.no_grad():
            obs_tensor = th.tensor(obs, dtype=th.float32)
            acts_tensor = th.tensor(acts, dtype=th.float32)
            next_obs_tensor = th.tensor(next_obs, dtype=th.float32)
            
            rewards = self.reward_net(obs_tensor, acts_tensor, next_obs_tensor, None)
            return rewards.cpu().numpy()
    
    def save_reward_net(self, path: str) -> None:
        """Save the trained reward network."""
        th.save(self.reward_net.state_dict(), path)
        self._logger.log(f"Saved reward network to {path}")
    
    def load_reward_net(self, path: str) -> None:
        """Load a trained reward network."""
        self.reward_net.load_state_dict(th.load(path))
        self._logger.log(f"Loaded reward network from {path}")
