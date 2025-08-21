"""Tests for NTRIL components."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from imitation.data import types
from imitation.scripts.NTRIL.noise_injection import NoiseInjector, GaussianActionNoiseInjector
from imitation.scripts.NTRIL.ranked_dataset import RankedDatasetBuilder
from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from imitation.scripts.NTRIL.utils import (
    analyze_ranking_quality,
    compute_trajectory_diversity,
    generate_noise_schedule,
)


class TestNoiseInjection:
    """Tests for noise injection components."""
    
    def test_gaussian_noise_injector_creation(self):
        """Test creation of Gaussian noise injector."""
        injector = NoiseInjector(strategy="gaussian_action")
        assert injector.strategy == "gaussian_action"
        assert isinstance(injector.injector, GaussianActionNoiseInjector)
    
    def test_noise_schedule_generation(self):
        """Test noise schedule generation."""
        injector = NoiseInjector()
        
        # Linear schedule
        linear_schedule = injector.get_noise_schedule(
            min_noise=0.0, max_noise=0.5, n_levels=5, schedule_type="linear"
        )
        assert len(linear_schedule) == 5
        assert linear_schedule[0] == 0.0
        assert linear_schedule[-1] == 0.5
        
        # Exponential schedule
        exp_schedule = injector.get_noise_schedule(
            min_noise=0.01, max_noise=0.5, n_levels=5, schedule_type="exponential"
        )
        assert len(exp_schedule) == 5
        assert exp_schedule[0] < exp_schedule[-1]
    
    @patch('imitation.scripts.NTRIL.noise_injection.policies')
    def test_noise_injection_mock(self, mock_policies):
        """Test noise injection with mock policy."""
        # Create mock policy
        mock_policy = Mock()
        mock_policy.observation_space = Mock()
        mock_policy.action_space = Mock()
        mock_policy.features_extractor_class = Mock()
        mock_policy.features_extractor_kwargs = {}
        mock_policy.normalize_images = False
        mock_policy.squash_output = False
        
        injector = NoiseInjector(strategy="gaussian_action")
        
        # Test injection doesn't crash
        noisy_policy = injector.inject_noise(mock_policy, noise_level=0.1)
        assert noisy_policy is not None


class TestRankedDataset:
    """Tests for ranked dataset builder."""
    
    def create_dummy_transitions(self, n_transitions: int, noise_level: float) -> types.Transitions:
        """Create dummy transitions for testing."""
        obs = np.random.randn(n_transitions, 4)
        acts = np.random.randn(n_transitions, 2)
        next_obs = np.random.randn(n_transitions, 4)
        dones = np.zeros(n_transitions, dtype=bool)
        infos = np.array([
            {"noise_level": noise_level, "ranking_score": noise_level}
            for _ in range(n_transitions)
        ])
        
        return types.Transitions(
            obs=obs,
            acts=acts,
            next_obs=next_obs,
            dones=dones,
            infos=infos,
        )
    
    def test_ranked_dataset_builder_creation(self):
        """Test creation of ranked dataset builder."""
        builder = RankedDatasetBuilder()
        assert builder.ranking_method == "noise_level"
    
    def test_build_ranked_dataset(self):
        """Test building ranked dataset."""
        builder = RankedDatasetBuilder()
        
        # Create dummy transition sets
        transitions_list = [
            self.create_dummy_transitions(10, 0.0),
            self.create_dummy_transitions(15, 0.2),
            self.create_dummy_transitions(12, 0.4),
        ]
        noise_rankings = [0.0, 0.2, 0.4]
        
        ranked_dataset = builder.build_ranked_dataset(transitions_list, noise_rankings)
        
        assert len(ranked_dataset.obs) == 10 + 15 + 12
        assert ranked_dataset.infos is not None
        assert len(ranked_dataset.infos) == 37
    
    def test_stratified_batches(self):
        """Test stratified batch creation."""
        builder = RankedDatasetBuilder()
        
        # Create dummy ranked dataset
        ranked_dataset = self.create_dummy_transitions(100, 0.2)
        
        batches = builder.create_ranked_batches(
            ranked_dataset,
            batch_size=16,
            ranking_batch_method="stratified",
        )
        
        assert len(batches) > 0
        assert all(len(batch.obs) <= 16 for batch in batches)
    
    def test_ranking_statistics(self):
        """Test ranking statistics computation."""
        builder = RankedDatasetBuilder()
        ranked_dataset = self.create_dummy_transitions(50, 0.3)
        
        stats = builder.get_ranking_statistics(ranked_dataset)
        
        assert "n_transitions" in stats
        assert "ranking_scores" in stats
        assert stats["n_transitions"] == 50


class TestRobustTubeMPC:
    """Tests for robust tube MPC."""
    
    def create_dummy_trajectory(self, length: int) -> types.Trajectory:
        """Create dummy trajectory for testing."""
        obs = np.random.randn(length + 1, 4)
        acts = np.random.randn(length, 2)
        rews = np.random.randn(length)
        
        return types.Trajectory(obs=obs, acts=acts, rews=rews, infos=None)
    
    def test_robust_tube_mpc_creation(self):
        """Test creation of robust tube MPC."""
        mpc = RobustTubeMPC(horizon=10, disturbance_bound=0.1)
        assert mpc.horizon == 10
        assert mpc.disturbance_bound == 0.1
    
    def test_dynamics_fitting(self):
        """Test fitting dynamics from trajectories."""
        mpc = RobustTubeMPC()
        
        # Create dummy trajectories
        trajectories = [
            self.create_dummy_trajectory(20)
            for _ in range(5)
        ]
        
        stats = mpc.fit_dynamics(trajectories)
        
        assert "mse" in stats
        assert "n_transitions" in stats
        assert mpc.state_dim is not None
        assert mpc.action_dim is not None
    
    def test_trajectory_augmentation(self):
        """Test trajectory augmentation."""
        mpc = RobustTubeMPC()
        
        # Fit dynamics first
        trajectories = [self.create_dummy_trajectory(20) for _ in range(3)]
        mpc.fit_dynamics(trajectories)
        
        # Test augmentation
        test_traj = self.create_dummy_trajectory(10)
        augmented = mpc.augment_trajectory(test_traj, noise_level=0.2, n_augmentations=3)
        
        assert len(augmented.obs) > 0
        assert len(augmented.acts) == len(augmented.obs)


class TestUtils:
    """Tests for utility functions."""
    
    def create_dummy_trajectory(self, length: int, total_reward: float) -> types.Trajectory:
        """Create dummy trajectory with specified total reward."""
        obs = np.random.randn(length + 1, 4)
        acts = np.random.randn(length, 2)
        rews = np.full(length, total_reward / length)  # Uniform reward distribution
        
        return types.Trajectory(obs=obs, acts=acts, rews=rews, infos=None)
    
    def test_generate_noise_schedule(self):
        """Test noise schedule generation."""
        schedule = generate_noise_schedule(
            min_noise=0.0, max_noise=0.5, n_levels=5, schedule_type="linear"
        )
        
        assert len(schedule) == 5
        assert schedule[0] == 0.0
        assert schedule[-1] == 0.5
        assert all(schedule[i] <= schedule[i+1] for i in range(len(schedule)-1))
    
    def test_compute_trajectory_diversity(self):
        """Test trajectory diversity computation."""
        trajectories = [
            self.create_dummy_trajectory(10, reward)
            for reward in [5.0, 10.0, 15.0]
        ]
        
        # Test return diversity
        return_diversity = compute_trajectory_diversity(trajectories, metric="return_diversity")
        assert return_diversity > 0
        
        # Test state diversity
        state_diversity = compute_trajectory_diversity(trajectories, metric="state_diversity")
        assert state_diversity >= 0
    
    def test_analyze_ranking_quality(self):
        """Test ranking quality analysis."""
        # Create dummy ranked dataset
        n_transitions = 100
        obs = np.random.randn(n_transitions, 4)
        acts = np.random.randn(n_transitions, 2)
        next_obs = np.random.randn(n_transitions, 4)
        dones = np.zeros(n_transitions, dtype=bool)
        
        # Create infos with ranking information
        infos = []
        for i in range(n_transitions):
            noise_level = np.random.uniform(0, 0.5)
            infos.append({
                "noise_level": noise_level,
                "ranking_score": noise_level,  # Perfect correlation
            })
        
        ranked_dataset = types.Transitions(
            obs=obs,
            acts=acts,
            next_obs=next_obs,
            dones=dones,
            infos=np.array(infos),
        )
        
        analysis = analyze_ranking_quality(ranked_dataset, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        assert "correlation_noise_ranking" in analysis
        assert "ranking_consistency_rate" in analysis
        assert analysis["correlation_noise_ranking"] > 0.8  # Should be high correlation


if __name__ == "__main__":
    # Run basic tests
    test_noise = TestNoiseInjection()
    test_noise.test_gaussian_noise_injector_creation()
    test_noise.test_noise_schedule_generation()
    
    test_dataset = TestRankedDataset()
    test_dataset.test_ranked_dataset_builder_creation()
    test_dataset.test_build_ranked_dataset()
    
    test_mpc = TestRobustTubeMPC()
    test_mpc.test_robust_tube_mpc_creation()
    test_mpc.test_dynamics_fitting()
    
    test_utils = TestUtils()
    test_utils.test_generate_noise_schedule()
    test_utils.test_compute_trajectory_diversity()
    
    print("All basic tests passed!")
