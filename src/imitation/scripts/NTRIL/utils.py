"""Utility functions for NTRIL pipeline."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from imitation.data import types


def visualize_noise_levels(
    rollouts_by_noise: List[List[types.Trajectory]],
    noise_levels: Sequence[float],
    save_path: Optional[str] = None,
) -> None:
    """Visualize trajectory statistics across different noise levels.
    
    Args:
        rollouts_by_noise: List of rollout lists for each noise level
        noise_levels: Corresponding noise levels
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('NTRIL Trajectory Analysis by Noise Level')
    
    # Compute statistics for each noise level
    mean_returns = []
    std_returns = []
    mean_lengths = []
    std_lengths = []
    
    for rollouts in rollouts_by_noise:
        returns = [sum(traj.rews) for traj in rollouts]
        lengths = [len(traj) for traj in rollouts]
        
        mean_returns.append(np.mean(returns))
        std_returns.append(np.std(returns))
        mean_lengths.append(np.mean(lengths))
        std_lengths.append(np.std(lengths))
    
    # Plot 1: Mean returns vs noise level
    axes[0, 0].errorbar(noise_levels, mean_returns, yerr=std_returns, 
                       marker='o', capsize=5)
    axes[0, 0].set_xlabel('Noise Level')
    axes[0, 0].set_ylabel('Episode Return')
    axes[0, 0].set_title('Episode Returns vs Noise Level')
    axes[0, 0].grid(True)
    
    # Plot 2: Mean episode lengths vs noise level
    axes[0, 1].errorbar(noise_levels, mean_lengths, yerr=std_lengths,
                       marker='s', capsize=5)
    axes[0, 1].set_xlabel('Noise Level')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Lengths vs Noise Level')
    axes[0, 1].grid(True)
    
    # Plot 3: Return distribution
    for i, (rollouts, noise_level) in enumerate(zip(rollouts_by_noise, noise_levels)):
        returns = [sum(traj.rews) for traj in rollouts]
        axes[1, 0].hist(returns, alpha=0.6, label=f'Noise {noise_level:.2f}', bins=15)
    axes[1, 0].set_xlabel('Episode Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Return Distributions')
    axes[1, 0].legend()
    
    # Plot 4: Number of rollouts per noise level
    n_rollouts = [len(rollouts) for rollouts in rollouts_by_noise]
    axes[1, 1].bar(range(len(noise_levels)), n_rollouts)
    axes[1, 1].set_xlabel('Noise Level Index')
    axes[1, 1].set_ylabel('Number of Rollouts')
    axes[1, 1].set_title('Rollout Count by Noise Level')
    axes[1, 1].set_xticks(range(len(noise_levels)))
    axes[1, 1].set_xticklabels([f'{nl:.2f}' for nl in noise_levels])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def analyze_ranking_quality(
    ranked_dataset: types.Transitions,
    noise_levels: Sequence[float],
) -> Dict[str, float]:
    """Analyze the quality of the ranking in the dataset.
    
    Args:
        ranked_dataset: Dataset with ranking information
        noise_levels: Original noise levels used
        
    Returns:
        Dictionary with ranking quality metrics
    """
    if not hasattr(ranked_dataset, 'infos') or ranked_dataset.infos is None:
        return {"error": "No ranking information available"}
    
    ranking_scores = [info["ranking_score"] for info in ranked_dataset.infos]
    noise_levels_data = [info["noise_level"] for info in ranked_dataset.infos]
    
    # Compute correlation between noise level and ranking score
    correlation = np.corrcoef(noise_levels_data, ranking_scores)[0, 1]
    
    # Compute ranking consistency (higher noise should have higher ranking scores)
    consistency_violations = 0
    total_comparisons = 0
    
    for i in range(len(ranking_scores)):
        for j in range(i + 1, len(ranking_scores)):
            noise_i, noise_j = noise_levels_data[i], noise_levels_data[j]
            score_i, score_j = ranking_scores[i], ranking_scores[j]
            
            # If noise_i < noise_j, then score_i should be < score_j
            if noise_i < noise_j and score_i >= score_j:
                consistency_violations += 1
            elif noise_i > noise_j and score_i <= score_j:
                consistency_violations += 1
            
            total_comparisons += 1
    
    consistency_rate = 1.0 - (consistency_violations / max(total_comparisons, 1))
    
    # Compute ranking entropy (how diverse are the rankings)
    unique_scores, counts = np.unique(ranking_scores, return_counts=True)
    probabilities = counts / len(ranking_scores)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
    
    return {
        "correlation_noise_ranking": correlation,
        "ranking_consistency_rate": consistency_rate,
        "ranking_entropy": entropy,
        "n_unique_rankings": len(unique_scores),
        "ranking_score_range": np.max(ranking_scores) - np.min(ranking_scores),
    }


def compute_trajectory_diversity(
    trajectories: List[types.Trajectory],
    metric: str = "state_diversity",
) -> float:
    """Compute diversity metric for a set of trajectories.
    
    Args:
        trajectories: List of trajectories
        metric: Diversity metric ("state_diversity", "action_diversity", "return_diversity")
        
    Returns:
        Diversity score
    """
    if not trajectories:
        return 0.0
    
    if metric == "state_diversity":
        # Compute average pairwise state distances
        all_states = np.concatenate([traj.obs for traj in trajectories])
        if len(all_states) < 2:
            return 0.0
        
        # Sample subset for efficiency
        n_sample = min(1000, len(all_states))
        sampled_states = all_states[np.random.choice(len(all_states), n_sample, replace=False)]
        
        distances = []
        for i in range(len(sampled_states)):
            for j in range(i + 1, len(sampled_states)):
                dist = np.linalg.norm(sampled_states[i] - sampled_states[j])
                distances.append(dist)
        
        return np.mean(distances)
    
    elif metric == "action_diversity":
        # Compute average pairwise action distances
        all_actions = np.concatenate([traj.acts for traj in trajectories])
        if len(all_actions) < 2:
            return 0.0
        
        # Sample subset for efficiency
        n_sample = min(1000, len(all_actions))
        sampled_actions = all_actions[np.random.choice(len(all_actions), n_sample, replace=False)]
        
        distances = []
        for i in range(len(sampled_actions)):
            for j in range(i + 1, len(sampled_actions)):
                dist = np.linalg.norm(sampled_actions[i] - sampled_actions[j])
                distances.append(dist)
        
        return np.mean(distances)
    
    elif metric == "return_diversity":
        # Compute standard deviation of returns
        returns = [sum(traj.rews) for traj in trajectories]
        return np.std(returns)
    
    else:
        raise ValueError(f"Unknown diversity metric: {metric}")


def generate_noise_schedule(
    min_noise: float = 0.0,
    max_noise: float = 0.5,
    n_levels: int = 6,
    schedule_type: str = "linear",
    curriculum: bool = False,
) -> List[float]:
    """Generate a noise schedule for NTRIL training.
    
    Args:
        min_noise: Minimum noise level
        max_noise: Maximum noise level
        n_levels: Number of noise levels
        schedule_type: Type of schedule ("linear", "exponential", "quadratic")
        curriculum: Whether to use curriculum learning (start with low noise)
        
    Returns:
        List of noise levels
    """
    if schedule_type == "linear":
        noise_levels = np.linspace(min_noise, max_noise, n_levels)
    elif schedule_type == "exponential":
        noise_levels = np.logspace(np.log10(max(min_noise, 1e-6)), np.log10(max_noise), n_levels)
    elif schedule_type == "quadratic":
        linear_vals = np.linspace(0, 1, n_levels)
        noise_levels = min_noise + (max_noise - min_noise) * linear_vals ** 2
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    noise_levels = noise_levels.tolist()
    
    if curriculum:
        # Sort from low to high noise for curriculum learning
        noise_levels.sort()
    
    return noise_levels


def create_ntril_report(
    training_stats: Dict,
    eval_stats: Dict,
    config: Dict,
    save_path: Optional[str] = None,
) -> str:
    """Create a comprehensive report of NTRIL training results.
    
    Args:
        training_stats: Training statistics
        eval_stats: Evaluation statistics  
        config: Training configuration
        save_path: Path to save the report
        
    Returns:
        Report as string
    """
    report_lines = [
        "# NTRIL Training Report",
        "",
        "## Configuration",
        f"- Noise levels: {config.get('noise_levels', 'N/A')}",
        f"- Rollouts per noise level: {config.get('n_rollouts_per_noise', 'N/A')}",
        f"- MPC horizon: {config.get('mpc_horizon', 'N/A')}",
        f"- Disturbance bound: {config.get('disturbance_bound', 'N/A')}",
        "",
        "## Training Results",
    ]
    
    # BC training results
    if "bc" in training_stats:
        bc_stats = training_stats["bc"]
        report_lines.extend([
            "### Behavioral Cloning",
            f"- Number of rollouts: {bc_stats.get('n_rollouts', 'N/A')}",
            f"- Mean return: {bc_stats.get('mean_return', 'N/A'):.3f}",
            f"- Mean length: {bc_stats.get('mean_length', 'N/A'):.1f}",
            "",
        ])
    
    # Rollout generation results
    if "rollouts" in training_stats:
        rollout_stats = training_stats["rollouts"]
        report_lines.extend([
            "### Noisy Rollout Generation",
            f"- Total rollouts: {rollout_stats.get('total_rollouts', 'N/A')}",
            f"- Rollouts per level: {rollout_stats.get('rollouts_per_level', 'N/A')}",
            "",
        ])
    
    # Data augmentation results
    if "augmentation" in training_stats:
        aug_stats = training_stats["augmentation"]
        report_lines.extend([
            "### Data Augmentation (Robust Tube MPC)",
            f"- Total augmented transitions: {aug_stats.get('total_augmented_transitions', 'N/A')}",
            f"- Augmentation ratio: {aug_stats.get('augmentation_ratio', 'N/A'):.2f}",
            "",
        ])
    
    # Ranking results
    if "ranking" in training_stats:
        ranking_stats = training_stats["ranking"]
        report_lines.extend([
            "### Ranked Dataset Construction",
            f"- Total ranked transitions: {ranking_stats.get('total_ranked_transitions', 'N/A')}",
            f"- Unique noise levels: {ranking_stats.get('unique_noise_levels', 'N/A')}",
            "",
        ])
    
    # IRL results
    if "irl" in training_stats:
        report_lines.extend([
            "### Demonstration Ranked IRL",
            "- IRL training completed successfully",
            "",
        ])
    
    # Evaluation results
    report_lines.extend([
        "## Evaluation Results",
        f"- Number of episodes: {eval_stats.get('n_episodes', 'N/A')}",
        f"- Mean return: {eval_stats.get('return_mean', 'N/A'):.3f}",
        f"- Return std: {eval_stats.get('return_std', 'N/A'):.3f}",
        f"- Mean length: {eval_stats.get('len_mean', 'N/A'):.1f}",
        "",
    ])
    
    report = "\n".join(report_lines)
    
    if save_path:
        Path(save_path).write_text(report)
        print(f"Saved report to {save_path}")
    
    return report
