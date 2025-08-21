"""Ranked dataset construction for NTRIL pipeline."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
from sklearn.preprocessing import LabelEncoder

from imitation.data import types
from imitation.util import util


class RankedDatasetBuilder:
    """Builds ranked datasets from trajectory data with noise level rankings."""
    
    def __init__(
        self,
        ranking_method: str = "noise_level",
        preference_noise: float = 0.1,
        min_preference_gap: float = 0.05,
    ):
        """Initialize ranked dataset builder.
        
        Args:
            ranking_method: Method for ranking ("noise_level", "performance", "hybrid")
            preference_noise: Noise level for preference generation
            min_preference_gap: Minimum gap between rankings for reliable preferences
        """
        self.ranking_method = ranking_method
        self.preference_noise = preference_noise
        self.min_preference_gap = min_preference_gap
        self.label_encoder = LabelEncoder()
    
    def build_ranked_dataset(
        self,
        transitions_list: List[types.Transitions],
        noise_rankings: List[float],
        performance_scores: Optional[List[float]] = None,
    ) -> types.Transitions:
        """Build a ranked dataset from multiple transition sets.
        
        Args:
            transitions_list: List of transition sets from different noise levels
            noise_rankings: Noise level for each transition set
            performance_scores: Optional performance scores for hybrid ranking
            
        Returns:
            Combined transitions with ranking information in infos
        """
        if len(transitions_list) != len(noise_rankings):
            raise ValueError("Number of transition sets must match number of noise rankings")
        
        # Combine all transitions
        all_obs = []
        all_acts = []
        all_next_obs = []
        all_dones = []
        all_infos = []
        all_rankings = []
        
        for i, (transitions, noise_level) in enumerate(zip(transitions_list, noise_rankings)):
            # Add transitions to combined dataset
            all_obs.extend(transitions.obs)
            all_acts.extend(transitions.acts)
            all_next_obs.extend(transitions.next_obs)
            all_dones.extend(transitions.dones)
            
            # Compute ranking score
            if self.ranking_method == "noise_level":
                ranking_score = noise_level
            elif self.ranking_method == "performance":
                if performance_scores is None:
                    raise ValueError("Performance scores required for performance ranking")
                ranking_score = -performance_scores[i]  # Lower performance = higher rank
            elif self.ranking_method == "hybrid":
                if performance_scores is None:
                    raise ValueError("Performance scores required for hybrid ranking")
                # Combine noise level and performance (both normalized)
                norm_noise = noise_level  # Assume already normalized [0, 1]
                norm_perf = -performance_scores[i]  # Negative for ranking
                ranking_score = 0.7 * norm_noise + 0.3 * norm_perf
            else:
                raise ValueError(f"Unknown ranking method: {self.ranking_method}")
            
            # Add ranking info to each transition
            for j in range(len(transitions.obs)):
                info = transitions.infos[j] if hasattr(transitions, 'infos') and transitions.infos is not None else {}
                info.update({
                    "noise_level": noise_level,
                    "ranking_score": ranking_score,
                    "dataset_index": i,
                })
                all_infos.append(info)
                all_rankings.append(ranking_score)
        
        # Convert to numpy arrays
        combined_transitions = types.Transitions(
            obs=np.array(all_obs),
            acts=np.array(all_acts),
            next_obs=np.array(all_next_obs),
            dones=np.array(all_dones),
            infos=np.array(all_infos),
        )
        
        return combined_transitions
    
    def generate_preference_pairs(
        self,
        transitions: types.Transitions,
        n_pairs: int = 1000,
        segment_length: int = 50,
    ) -> List[Tuple[types.Transitions, types.Transitions, int]]:
        """Generate preference pairs from ranked transitions.
        
        Args:
            transitions: Ranked transitions dataset
            n_pairs: Number of preference pairs to generate
            segment_length: Length of trajectory segments for comparison
            
        Returns:
            List of (segment_1, segment_2, preference) tuples
            preference: 0 if segment_1 preferred, 1 if segment_2 preferred
        """
        if not hasattr(transitions, 'infos') or transitions.infos is None:
            raise ValueError("Transitions must have ranking info")
        
        preference_pairs = []
        n_transitions = len(transitions.obs)
        
        for _ in range(n_pairs):
            # Sample two segments
            seg1_start = np.random.randint(0, n_transitions - segment_length)
            seg2_start = np.random.randint(0, n_transitions - segment_length)
            
            # Extract segments
            seg1 = self._extract_segment(transitions, seg1_start, segment_length)
            seg2 = self._extract_segment(transitions, seg2_start, segment_length)
            
            # Get ranking scores
            score1 = np.mean([info["ranking_score"] for info in seg1.infos])
            score2 = np.mean([info["ranking_score"] for info in seg2.infos])
            
            # Only create pair if there's sufficient ranking difference
            if abs(score1 - score2) < self.min_preference_gap:
                continue
            
            # Determine preference (lower ranking score is better)
            # Add preference noise
            noisy_score1 = score1 + np.random.normal(0, self.preference_noise)
            noisy_score2 = score2 + np.random.normal(0, self.preference_noise)
            
            preference = 0 if noisy_score1 < noisy_score2 else 1
            
            preference_pairs.append((seg1, seg2, preference))
        
        return preference_pairs
    
    def _extract_segment(
        self,
        transitions: types.Transitions,
        start_idx: int,
        length: int,
    ) -> types.Transitions:
        """Extract a segment from transitions."""
        end_idx = start_idx + length
        
        return types.Transitions(
            obs=transitions.obs[start_idx:end_idx],
            acts=transitions.acts[start_idx:end_idx],
            next_obs=transitions.next_obs[start_idx:end_idx],
            dones=transitions.dones[start_idx:end_idx],
            infos=transitions.infos[start_idx:end_idx] if transitions.infos is not None else None,
        )
    
    def create_ranked_batches(
        self,
        transitions: types.Transitions,
        batch_size: int = 32,
        ranking_batch_method: str = "stratified",
    ) -> List[types.Transitions]:
        """Create batches with ranking-aware sampling.
        
        Args:
            transitions: Ranked transitions dataset
            batch_size: Size of each batch
            ranking_batch_method: Method for creating batches ("stratified", "uniform", "weighted")
            
        Returns:
            List of transition batches
        """
        if not hasattr(transitions, 'infos') or transitions.infos is None:
            raise ValueError("Transitions must have ranking info")
        
        n_transitions = len(transitions.obs)
        ranking_scores = np.array([info["ranking_score"] for info in transitions.infos])
        
        if ranking_batch_method == "stratified":
            # Ensure each batch has samples from different ranking levels
            batches = self._create_stratified_batches(
                transitions, ranking_scores, batch_size
            )
        elif ranking_batch_method == "uniform":
            # Uniform random sampling
            batches = self._create_uniform_batches(transitions, batch_size)
        elif ranking_batch_method == "weighted":
            # Weighted sampling based on ranking diversity
            batches = self._create_weighted_batches(
                transitions, ranking_scores, batch_size
            )
        else:
            raise ValueError(f"Unknown ranking batch method: {ranking_batch_method}")
        
        return batches
    
    def _create_stratified_batches(
        self,
        transitions: types.Transitions,
        ranking_scores: np.ndarray,
        batch_size: int,
    ) -> List[types.Transitions]:
        """Create stratified batches ensuring ranking diversity."""
        # Divide ranking scores into quantiles
        n_quantiles = min(4, batch_size // 2)  # At least 2 samples per quantile
        quantiles = np.percentile(ranking_scores, np.linspace(0, 100, n_quantiles + 1))
        
        # Assign each transition to a quantile
        quantile_indices = []
        for i in range(n_quantiles):
            if i == n_quantiles - 1:
                # Last quantile includes maximum value
                mask = (ranking_scores >= quantiles[i]) & (ranking_scores <= quantiles[i + 1])
            else:
                mask = (ranking_scores >= quantiles[i]) & (ranking_scores < quantiles[i + 1])
            quantile_indices.append(np.where(mask)[0])
        
        batches = []
        n_transitions = len(transitions.obs)
        n_batches = n_transitions // batch_size
        
        for _ in range(n_batches):
            batch_indices = []
            
            # Sample from each quantile
            samples_per_quantile = batch_size // n_quantiles
            remaining_samples = batch_size % n_quantiles
            
            for i, indices in enumerate(quantile_indices):
                if len(indices) == 0:
                    continue
                
                n_samples = samples_per_quantile
                if i < remaining_samples:
                    n_samples += 1
                
                if n_samples > 0:
                    sampled_indices = np.random.choice(
                        indices, size=min(n_samples, len(indices)), replace=True
                    )
                    batch_indices.extend(sampled_indices)
            
            # Create batch
            if batch_indices:
                batch = types.Transitions(
                    obs=transitions.obs[batch_indices],
                    acts=transitions.acts[batch_indices],
                    next_obs=transitions.next_obs[batch_indices],
                    dones=transitions.dones[batch_indices],
                    infos=transitions.infos[batch_indices] if transitions.infos is not None else None,
                )
                batches.append(batch)
        
        return batches
    
    def _create_uniform_batches(
        self,
        transitions: types.Transitions,
        batch_size: int,
    ) -> List[types.Transitions]:
        """Create uniform random batches."""
        batches = []
        n_transitions = len(transitions.obs)
        n_batches = n_transitions // batch_size
        
        # Shuffle indices
        indices = np.random.permutation(n_transitions)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            batch = types.Transitions(
                obs=transitions.obs[batch_indices],
                acts=transitions.acts[batch_indices],
                next_obs=transitions.next_obs[batch_indices],
                dones=transitions.dones[batch_indices],
                infos=transitions.infos[batch_indices] if transitions.infos is not None else None,
            )
            batches.append(batch)
        
        return batches
    
    def _create_weighted_batches(
        self,
        transitions: types.Transitions,
        ranking_scores: np.ndarray,
        batch_size: int,
    ) -> List[types.Transitions]:
        """Create weighted batches based on ranking diversity."""
        # Compute sampling weights (higher weight for diverse rankings)
        score_std = np.std(ranking_scores)
        if score_std == 0:
            weights = np.ones(len(ranking_scores))
        else:
            # Weight based on distance from mean
            weights = np.abs(ranking_scores - np.mean(ranking_scores)) / score_std
            weights = weights + 0.1  # Ensure minimum weight
        
        weights /= np.sum(weights)
        
        batches = []
        n_transitions = len(transitions.obs)
        n_batches = n_transitions // batch_size
        
        for _ in range(n_batches):
            # Sample with replacement based on weights
            batch_indices = np.random.choice(
                n_transitions, size=batch_size, replace=True, p=weights
            )
            
            batch = types.Transitions(
                obs=transitions.obs[batch_indices],
                acts=transitions.acts[batch_indices],
                next_obs=transitions.next_obs[batch_indices],
                dones=transitions.dones[batch_indices],
                infos=transitions.infos[batch_indices] if transitions.infos is not None else None,
            )
            batches.append(batch)
        
        return batches
    
    def get_ranking_statistics(
        self,
        transitions: types.Transitions,
    ) -> Dict[str, Any]:
        """Compute statistics about the ranking distribution."""
        if not hasattr(transitions, 'infos') or transitions.infos is None:
            return {}
        
        ranking_scores = [info["ranking_score"] for info in transitions.infos]
        noise_levels = [info["noise_level"] for info in transitions.infos]
        
        stats = {
            "n_transitions": len(transitions.obs),
            "ranking_scores": {
                "mean": np.mean(ranking_scores),
                "std": np.std(ranking_scores),
                "min": np.min(ranking_scores),
                "max": np.max(ranking_scores),
                "median": np.median(ranking_scores),
            },
            "noise_levels": {
                "unique": len(np.unique(noise_levels)),
                "mean": np.mean(noise_levels),
                "std": np.std(noise_levels),
                "min": np.min(noise_levels),
                "max": np.max(noise_levels),
            },
        }
        
        # Distribution of samples per ranking level
        unique_scores, counts = np.unique(ranking_scores, return_counts=True)
        stats["ranking_distribution"] = dict(zip(unique_scores, counts))
        
        return stats
