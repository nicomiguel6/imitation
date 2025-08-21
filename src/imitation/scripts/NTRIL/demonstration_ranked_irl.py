"""Demonstration Ranked Inverse Reinforcement Learning for NTRIL pipeline."""

import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
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
        demonstrations: Sequence[types.Trajectory],
        ranked_transitions: types.Transitions,
        segment_length: int = 50,
    ):
        """Initialize ranked transitions dataset.
        
        Args:
            demonstrations: Expert demonstration trajectories
            ranked_transitions: Transitions with ranking information
            segment_length: Length of trajectory segments for comparison
        """
        self.demonstrations = demonstrations
        self.ranked_transitions = ranked_transitions
        self.segment_length = segment_length
        
        # Prepare expert segments
        self.expert_segments = self._extract_expert_segments()
        
        # Prepare ranked segments
        self.ranked_segments = self._extract_ranked_segments()
    
    def _extract_expert_segments(self) -> List[types.Transitions]:
        """Extract segments from expert demonstrations."""
        segments = []
        
        for traj in self.demonstrations:
            for start_idx in range(len(traj.obs) - self.segment_length + 1):
                end_idx = start_idx + self.segment_length
                
                segment = types.Transitions(
                    obs=traj.obs[start_idx:end_idx],
                    acts=traj.acts[start_idx:end_idx],
                    next_obs=traj.obs[start_idx + 1:end_idx + 1],
                    dones=np.zeros(self.segment_length, dtype=bool),  # Assume not done mid-trajectory
                    infos=np.array([{"expert": True, "ranking_score": 0.0}] * self.segment_length),
                )
                segments.append(segment)
        
        return segments
    
    def _extract_ranked_segments(self) -> List[types.Transitions]:
        """Extract segments from ranked transitions."""
        segments = []
        n_transitions = len(self.ranked_transitions.obs)
        
        for start_idx in range(0, n_transitions - self.segment_length + 1, self.segment_length // 2):
            end_idx = start_idx + self.segment_length
            if end_idx > n_transitions:
                break
            
            segment = types.Transitions(
                obs=self.ranked_transitions.obs[start_idx:end_idx],
                acts=self.ranked_transitions.acts[start_idx:end_idx],
                next_obs=self.ranked_transitions.next_obs[start_idx:end_idx],
                dones=self.ranked_transitions.dones[start_idx:end_idx],
                infos=self.ranked_transitions.infos[start_idx:end_idx] if self.ranked_transitions.infos is not None else None,
            )
            segments.append(segment)
        
        return segments
    
    def __len__(self) -> int:
        """Return number of possible segment pairs."""
        return len(self.expert_segments) * len(self.ranked_segments)
    
    def __getitem__(self, idx: int) -> Tuple[types.Transitions, types.Transitions, float]:
        """Get a segment pair with preference label."""
        expert_idx = idx % len(self.expert_segments)
        ranked_idx = idx // len(self.expert_segments)
        
        if ranked_idx >= len(self.ranked_segments):
            ranked_idx = ranked_idx % len(self.ranked_segments)
        
        expert_segment = self.expert_segments[expert_idx]
        ranked_segment = self.ranked_segments[ranked_idx]
        
        # Expert demonstrations are always preferred (ranking score 0.0)
        # Higher ranking scores indicate worse performance
        ranked_score = np.mean([
            info["ranking_score"] for info in ranked_segment.infos
        ]) if ranked_segment.infos is not None else 0.5
        
        # Preference: 0 if expert preferred, 1 if ranked segment preferred
        # Since experts should always be preferred, this is always 0
        # unless the ranked segment has very low noise (high quality)
        preference = 0.0 if ranked_score > 0.1 else 1.0
        
        return expert_segment, ranked_segment, preference


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
