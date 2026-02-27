"""Demonstration Ranked Inverse Reinforcement Learning for NTRIL pipeline."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
import random
import torch.nn as nn
from stable_baselines3.common import vec_env
from torch.utils.data import DataLoader, Dataset

from imitation.algorithms import base
from imitation.data import types
from imitation.rewards import reward_nets
from imitation.util import logger as imit_logger


class RankedTransitionsDataset(Dataset):
    """PyTorch dataset for ranked transitions."""

    def __init__(
        self,
        demonstrations: Optional[List[Sequence[types.TrajectoryWithRew]]] = None,
        training_samples: Optional[List[Tuple[Tuple[np.ndarray, np.ndarray], int]]] = None,
        num_snippets: int = 10,
        min_segment_length: int = 50,
        max_segment_length: int = 100,
    ):
        """Initialize ranked transitions dataset.

        From the total set of demonstrations, a number of snippets will be generated,
        unless precomputed training_samples are provided.

        Args:
            demonstrations: Noisy rollout demonstration trajectories.
            training_samples: Optional precomputed samples of the form
                ((snippet_i, snippet_j), label). If provided, no new samples
                are generated from demonstrations.
            num_snippets: Total number of training samples to generate when
                building from demonstrations.
            min_segment_length: Minimum length of segments extracted from a trajectory.
            max_segment_length: Maximum length of segments extracted from a trajectory.
        """
        if demonstrations is None and training_samples is None:
            raise ValueError("Either demonstrations or training samples must be provided")

        self.num_snippets = num_snippets
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.demo_dict = {}
        self.demonstrations = []
        self.training_data = {"traj": [], "label": []}

        if training_samples is not None:
            # Directly load precomputed samples.
            self.load_training_samples(training_samples)
        elif demonstrations is not None:
            # Build from raw demonstrations.
            self._append_demonstrations(demonstrations)
            self._build_dict()
            self._generate_training_samples()

        # Prepare expert segments
        # self.expert_segments = self._extract_expert_segments()

        # Prepare ranked segments
        # self.ranked_segments = self._extract_ranked_segments()

    def _append_demonstrations(
        self, demonstrations: List[Sequence[types.TrajectoryWithRew]]
    ):
        """Store all demonstrations"""
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

        Preference pairs are labeled purely by noise level: lower noise epsilon
        is assumed to be better. Start indices are sampled independently per
        trajectory, so no temporal ordering is imposed across noise bins.
        """
        step = 2
        noise_levels = list(self.demo_dict.keys())

        for _ in range(self.num_snippets):
            # Pick two distinct noise levels at random.
            ni, nj = 0, 0
            while ni == nj:
                ni = np.random.randint(len(noise_levels))
                nj = np.random.randint(len(noise_levels))

            # Pick a random trajectory from each noise bin.
            ti = random.choice(self.demo_dict[noise_levels[ni]])
            tj = random.choice(self.demo_dict[noise_levels[nj]])

            # Sample a shared segment length, capped to whichever trajectory is shorter.
            if self.min_segment_length == self.max_segment_length:
                rand_length = self.min_segment_length
            else:
                rand_length = np.random.randint(
                    self.min_segment_length, self.max_segment_length
                )
            rand_length = min(rand_length, len(ti.obs), len(tj.obs))
            if rand_length < 1:
                raise ValueError(
                    f"Segment length {rand_length} is invalid for trajectories "
                    f"of lengths {len(ti.obs)} and {len(tj.obs)}."
                )

            # Sample start indices independently — no temporal ordering imposed.
            ti_start = np.random.randint(len(ti.obs) - rand_length + 1)
            tj_start = np.random.randint(len(tj.obs) - rand_length + 1)

            snip_i = ti.obs[ti_start : ti_start + rand_length : step]
            snip_j = tj.obs[tj_start : tj_start + rand_length : step]

            # label=0 → J(τ_i) should win (i is better); label=1 → J(τ_j) should win.
            # Lower noise = better trajectory, so assign label=0 when i has lower noise.
            label = 0 if noise_levels[ni] < noise_levels[nj] else 1

            self.training_data["traj"].append((snip_i, snip_j))
            self.training_data["label"].append(label)

        # save all data to file


    def load_training_samples(
        self, training_samples: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]
    ):
        """Load supplied training samples into dict format

        Args:
            training_samples (List[Tuple[Tuple[np.ndarray, np.ndarray], int]]): Training samples given as:
            (([traj_snippet_i], [traj_snippet_2]), label)
        """
        for training_sample in training_samples:
            self.training_data["traj"].append(training_sample[0])
            self.training_data["label"].append(training_sample[1])

    def __len__(self) -> int:
        """Return number of possible segment pairs."""
        return len(self.training_data["label"])

    def __getitem__(
        self, idx: int
    ) -> Tuple[types.Transitions, types.Transitions, float]:
        """Get a specific trajectory snippet pairand label at index idx"""
        return self.training_data["traj"][idx], self.training_data["label"][idx]


class DemonstrationRankedIRL(base.BaseImitationAlgorithm):
    """Demonstration Ranked Inverse Reinforcement Learning.

    This class learns reward functions from ranked demonstrations and
    augmented trajectory data, using the ranking structure to improve
    reward learning.
    """

    def __init__(
        self,
        reward_net: reward_nets.TrajectoryRewardNet,
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
        device: th.device = "mps",
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
        self.device = device

        # Move reward network to device
        self.reward_net = self.reward_net.to(device)

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
        train_dataloader: DataLoader,
        n_epochs: int = 100,
        eval_interval: int = 1,
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
        # Create dataloader
        self.train_dataloader = train_dataloader

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

                # # Evaluate reward network
                # eval_stats = self._evaluate_reward_net(demonstrations, ranked_dataset)
                # for key, value in eval_stats.items():
                #     if key not in stats:
                #         stats[key] = []
                #     stats[key].append(value)

        return stats

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        if self.train_dataloader is None:
            raise ValueError("Dataloader not initialized")

        total_loss = 0.0
        n_batches = 0

        self.reward_net.train()

        for batch in self.train_dataloader:
            segment_pairs, label = batch

            self.optimizer.zero_grad()

            # Forward pass
            loss = self._compute_loss(segment_pairs, label)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_loss(
        self,
        segment_pairs: List[Tuple[np.ndarray, np.ndarray]],
        labels: List[int],
    ) -> th.Tensor:
        """Compute the total loss for a batch."""

        batch_outputs = []
        batch_sum_abs_rewards = []
        loss_criterion = nn.CrossEntropyLoss()

        # Iterate over each segment pair and label
        for segment_pair, _ in zip(segment_pairs, labels):

            # convert to torch tensor
            traj_i = th.from_numpy(segment_pair[0]).to(self.device)
            traj_j = th.from_numpy(segment_pair[1]).to(self.device)

            # ensure both are float32
            traj_i = traj_i.float()
            traj_j = traj_j.float()

            # calculate cumulative reward
            outputs, sum_abs_rewards = self.reward_net.compare(traj_i, traj_j)

            # append to collection
            batch_outputs.append(outputs)
            batch_sum_abs_rewards.append(sum_abs_rewards)

        # Stack into tensors
        batch_outputs = th.stack(batch_outputs, dim=0)
        batch_sum_abs_rewards = th.stack(batch_sum_abs_rewards, dim=0)
        batch_labels = th.tensor(labels, dtype=th.long, device=self.device)

        # Preference loss (Bradley-Terry model)
        preference_loss = loss_criterion(batch_outputs, batch_labels)

        # Regularization loss
        reg_loss = batch_sum_abs_rewards.mean()

        # Total loss
        total_loss = (
            self.preference_loss_weight * preference_loss
            + self.regularization_weight * reg_loss
        )

        return total_loss

    def _collate_fn(
        self, batch
    ) -> Tuple[List[types.Transitions], List[types.Transitions], th.Tensor]:
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
            indices = np.random.choice(
                len(ranked_dataset.obs) - 1, n_samples, replace=False
            )

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
