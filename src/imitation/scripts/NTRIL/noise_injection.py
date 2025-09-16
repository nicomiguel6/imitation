"""Noise injection module for NTRIL pipeline."""

import abc
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common import policies

from imitation.util import util


class BaseNoiseInjector(abc.ABC):
    """Abstract base class for noise injection strategies."""
    
    @abc.abstractmethod
    def inject_noise(
        self, 
        policy: policies.BasePolicy, 
        noise_level: float,
    ) -> policies.BasePolicy:
        """Inject noise into a policy.
        
        Args:
            policy: The policy to add noise to
            noise_level: The level of noise to inject (typically 0.0 to 1.0)
            
        Returns:
            A new policy with noise injected
        """
        # create noise injector


        return NoisyPolicy(policy, )


class GaussianActionNoiseInjector(BaseNoiseInjector):
    """Injects Gaussian noise into policy actions."""
    
    def __init__(
        self, 
        noise_std_scale: float = 1.0,
        clip_actions: bool = True,
    ):
        """Initialize Gaussian action noise injector.
        
        Args:
            noise_std_scale: Scaling factor for noise standard deviation
            clip_actions: Whether to clip actions to action space bounds
        """
        self.noise_std_scale = noise_std_scale
        self.clip_actions = clip_actions
    
    def inject_noise(
        self, 
        policy: policies.BasePolicy, 
        noise_level: float,
    ) -> policies.BasePolicy:
        """Inject Gaussian noise into policy actions."""
        return NoisyPolicy(
            base_policy=policy,
            noise_injector=self,
            noise_level=noise_level,
        )


class UniformActionNoiseInjector(BaseNoiseInjector):
    """Injects uniform noise into policy actions."""
    
    def __init__(
        self, 
        noise_range_scale: float = 1.0,
        clip_actions: bool = True,
    ):
        """Initialize uniform action noise injector.
        
        Args:
            noise_range_scale: Scaling factor for noise range
            clip_actions: Whether to clip actions to action space bounds
        """
        self.noise_range_scale = noise_range_scale
        self.clip_actions = clip_actions
    
    def inject_noise(
        self, 
        policy: policies.BasePolicy, 
        noise_level: float,
    ) -> policies.BasePolicy:
        """Inject uniform noise into policy actions."""
        return NoisyPolicy(
            base_policy=policy,
            noise_injector=self,
            noise_level=noise_level,
        )


class ParameterNoiseInjector(BaseNoiseInjector):
    """Injects noise into policy parameters."""
    
    def __init__(self, noise_std_scale: float = 1.0):
        """Initialize parameter noise injector.
        
        Args:
            noise_std_scale: Scaling factor for parameter noise standard deviation
        """
        self.noise_std_scale = noise_std_scale
    
    def inject_noise(
        self, 
        policy: policies.BasePolicy, 
        noise_level: float,
    ) -> policies.BasePolicy:
        """Inject noise into policy parameters."""
        # Create a copy of the policy
        noisy_policy = type(policy)(
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            lr_schedule=lambda _: 0,  # Not used for inference
        )
        
        # Copy the state dict and add noise
        state_dict = policy.state_dict()
        for name, param in state_dict.items():
            if param.dtype.is_floating_point:
                noise = th.randn_like(param) * noise_level * self.noise_std_scale
                state_dict[name] = param + noise
        
        noisy_policy.load_state_dict(state_dict)
        return noisy_policy
    
class EpsilonGreedyNoiseInjector(BaseNoiseInjector):
    """Injects epsilon-greedy noise into policy actions."""

    def inject_noise(
        self,
        policy: policies.BasePolicy,
        noise_level: float,
    ) -> policies.BasePolicy:
        """Inject epsilon-greedy noise into policy actions."""
        return NoisyPolicy(
            base_policy=policy,
            noise_injector=self,
            noise_level=noise_level,
        )

    def apply_noise(
        self,
        base_action: th.Tensor,
        action_space: gym.Space,
        noise_level: float,
    ) -> th.Tensor:
        """With probability epsilon, return a random action; else, return base_action."""
        if np.random.rand() < noise_level:
            if isinstance(action_space, gym.spaces.Box):
                random_action = th.from_numpy(
                    np.random.uniform(
                        low=action_space.low,
                        high=action_space.high,
                        size=base_action.shape
                    )
                ).to(base_action.device, dtype=base_action.dtype)
            elif isinstance(action_space, gym.spaces.Discrete):
                random_action = th.tensor(
                    np.random.randint(action_space.n, size=base_action.shape),
                    device=base_action.device,
                    dtype=base_action.dtype,
                )
            else:
                raise NotImplementedError("Epsilon-greedy not implemented for this action space.")
            return random_action
        return base_action


class NoisyPolicy(policies.BasePolicy):
    """Wrapper policy that adds noise to actions from a base policy."""
    
    def __init__(
        self,
        base_policy: policies.BasePolicy,
        noise_injector: BaseNoiseInjector,
        noise_level: float,
    ):
        """Initialize noisy policy wrapper.
        
        Args:
            base_policy: The base policy to wrap
            noise_injector: The noise injection strategy
            noise_level: The level of noise to inject
        """
        super().__init__(
            observation_space=base_policy.observation_space,
            action_space=base_policy.action_space,
            features_extractor_class=base_policy.features_extractor_class,
            features_extractor_kwargs=base_policy.features_extractor_kwargs,
            normalize_images=base_policy.normalize_images,
            squash_output=base_policy.squash_output,
        )
        
        self.base_policy = base_policy
        self.noise_injector = noise_injector
        self.noise_level = noise_level
    
    def _predict(
        self, 
        observation: th.Tensor, 
        deterministic: bool = False,
    ) -> th.Tensor:
        """Predict action with noise injection."""
        base_action = self.base_policy._predict(observation, deterministic)

        if self.noise_level == 0.0 or deterministic or self.noise_injector is None:
            return base_action

        # Use the noise injector's apply_noise method if it exists
        if hasattr(self.noise_injector, "apply_noise"):
            return self.noise_injector.apply_noise(
                base_action, self.action_space, self.noise_level
            )
        else:
            # fallback: return base_action (or implement other noise logic)
            return base_action
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Forward pass through the policy."""
        return self._predict(obs, deterministic)


class NoiseInjector:
    """Main noise injector class that provides different noise injection strategies."""
    
    def __init__(
        self,
        strategy: str = "gaussian_action",
        **kwargs,
    ):
        """Initialize noise injector.
        
        Args:
            strategy: Noise injection strategy ("gaussian_action", "uniform_action", "parameter")
            **kwargs: Additional arguments for the specific strategy
        """
        self.strategy = strategy
        
        if strategy == "gaussian_action":
            self.injector = GaussianActionNoiseInjector(**kwargs)
        elif strategy == "uniform_action":
            self.injector = UniformActionNoiseInjector(**kwargs)
        elif strategy == "parameter":
            self.injector = ParameterNoiseInjector(**kwargs)
        else:
            raise ValueError(f"Unknown noise injection strategy: {strategy}")
    
    def inject_noise(
        self, 
        policy: policies.BasePolicy, 
        noise_level: float,
    ) -> policies.BasePolicy:
        """Inject noise into a policy.
        
        Args:
            policy: The policy to add noise to
            noise_level: The level of noise to inject
            
        Returns:
            A policy with noise injected
        """
        return self.injector.inject_noise(policy, noise_level)
    
    def get_noise_schedule(
        self,
        min_noise: float = 0.0,
        max_noise: float = 0.5,
        n_levels: int = 6,
        schedule_type: str = "linear",
    ) -> np.ndarray:
        """Generate a noise schedule.
        
        Args:
            min_noise: Minimum noise level
            max_noise: Maximum noise level
            n_levels: Number of noise levels
            schedule_type: Type of schedule ("linear", "exponential", "quadratic")
            
        Returns:
            Array of noise levels
        """
        if schedule_type == "linear":
            return np.linspace(min_noise, max_noise, n_levels)
        elif schedule_type == "exponential":
            return np.logspace(np.log10(max(min_noise, 1e-6)), np.log10(max_noise), n_levels)
        elif schedule_type == "quadratic":
            linear_vals = np.linspace(0, 1, n_levels)
            return min_noise + (max_noise - min_noise) * linear_vals ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
