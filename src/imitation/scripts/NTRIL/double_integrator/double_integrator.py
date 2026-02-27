"""Double Integrator Gymnasium Environment.

A double integrator is a classic control system where:
- State: [position, velocity]
- Action: [acceleration]
- Dynamics:
    position_ddot = acceleration
"""

from typing import Dict, Optional, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from imitation.policies.base import NonTrainablePolicy
from scipy.linalg import solve_discrete_are


class DoubleIntegratorEnv(gym.Env):
    """Double integrator environment for control tasks.

    The agent controls a double integrator system to reach a target position.
    The state consists of position and velocity, and the action is acceleration.

    Dynamics:
        x = [position, velocity]
        u = [acceleration]

        position_ddot = acceleration

        A = [[0, 1],
             [0, 0]]
        B = [[0],
             [1]]

    where position is the system position, velocity is the system velocity, and
    acceleration is the control input.
    """

    def __init__(
        self,
        dt: float = 1.0,
        max_position: float = 50.0,
        max_velocity: float = 50.0,
        max_acceleration: float = 20.0,
        target_position: float = 0.0,
        position_tolerance: float = 0.1,
        velocity_tolerance: float = 0.1,
        position_cost_weight: float = 10.0,
        velocity_cost_weight: float = 1.0,
        control_cost_weight: float = 0.1,
        max_episode_steps: int = 200,
    ):
        """Initialize the double integrator environment.

        Args:
            dt: Time step for integration
            max_position: Maximum absolute position (bounds for initial state)
            max_velocity: Maximum absolute velocity (bounds for initial state)
            max_acceleration: Maximum absolute acceleration (action bounds)
            target_position: Target position to reach
            position_tolerance: Tolerance for reaching target position
            velocity_tolerance: Tolerance for reaching target velocity
            position_cost_weight: Weight for position error in reward
            velocity_cost_weight: Weight for velocity error in reward
            control_cost_weight: Weight for control effort in reward
            max_episode_steps: Maximum number of steps per episode
        """
        super().__init__()

        self.dt = dt
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.target_position = target_position
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.position_cost_weight = position_cost_weight
        self.velocity_cost_weight = velocity_cost_weight
        self.control_cost_weight = control_cost_weight
        self.max_episode_steps = max_episode_steps

        # Observation space: [position, velocity]
        self.observation_space = Box(
            low=np.array(
                [-max_position, -max_velocity],
                dtype=np.float32,
            ),
            high=np.array(
                [max_position, max_velocity],
                dtype=np.float32,
            ),
        )

        # Action space: [acceleration]
        self.action_space = Box(
            low=np.array([-max_acceleration], dtype=np.float32),
            high=np.array([max_acceleration], dtype=np.float32),
        )

        # State: [position, velocity]
        self.state = None
        self.step_count = 0

        # Dynamics
        self.A = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.B = np.array([[0.0], [1.0]])

        # Cost matrices
        self.Q = np.diag([position_cost_weight, velocity_cost_weight])
        self.R = np.diag([control_cost_weight])
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)

    def reset(
        self,
        state: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional dictionary with reset options

        Returns:
            observation: Initial observation [position, velocity]
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        # Sample initial state uniformly from bounds
        if state is None:
            self.state = self.np_random.uniform(
                low=[-5.0, -5.0],
                high=[5.0, 5.0],
            ).astype(np.float32)
        else:
            self.state = state.astype(np.float32)

        self.step_count = 0

        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment forward one time step.

        Args:
            action: Acceleration control input [acceleration]

        Returns:
            observation: Next observation [position, velocity]
            reward: Reward for this step
            terminated: Whether episode terminated (reached target)
            truncated: Whether episode was truncated (max steps)
            info: Dictionary with additional information
        """
        # Clip action to valid range
        action = np.asarray(action, dtype=np.float32).reshape(1,)
        action = np.clip(action, -self.max_acceleration, self.max_acceleration)

        xdot = self.A @ self.state + self.B @ action
        self.state = self.state + xdot * self.dt
        
        position = self.state[0]
        velocity = self.state[1]

        # velocity += action * self.dt
        if velocity > self.max_velocity:
            velocity = self.max_velocity
        if velocity < -self.max_velocity:
            velocity = -self.max_velocity
        # position += velocity * self.dt
        if position > self.max_position:
            position = self.max_position
        if position < -self.max_position:
            position = -self.max_position
        if position == -self.max_position and velocity < 0:
            velocity = 0
        if position == self.max_position and velocity > 0:
            velocity = 0

        # Important: write clipped values back to internal state.
        # Otherwise self.state can silently diverge even though the returned values look bounded.
        self.state = np.array([position, velocity], dtype=np.float32)
        # self.state = self.state.astype(np.float32)
        self.step_count += 1

        # Compute reward (negative cost)
        # Target is at target_position with zero velocity
        position_error = position - self.target_position
        velocity_error = velocity  # Target velocity is 0

        # Position and velocity errors squared
        position_error_sq = position_error**2
        velocity_error_sq = velocity_error**2
        # Control effort
        control_effort = action[0] ** 2

        reward = -(
            self.position_cost_weight * position_error_sq
            + self.velocity_cost_weight * velocity_error_sq
            + self.control_cost_weight * control_effort
        )
        # reward = -self.state.T @ self.Q @ self.state - action.T @ self.R @ action


        # Episodes terminate only on max steps.
        terminated = False
        
        # Add terminal cost given by D.Algebraic Riccati Equation (ARE) 
        truncated = self.step_count >= self.max_episode_steps
        if truncated:
            terminal_cost = self.state.T @ self.P @ self.state
            reward -= terminal_cost.item()

        reward = reward.astype(np.float32)
        # Ensure returned state is float32
        self.state = self.state.astype(np.float32)

        info = {
            "position": position,
            "velocity": velocity,
            "position_error": position_error,
            "velocity_error": velocity_error,
            "target_reached": False,
        }

        return self.state.copy(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment (optional, not implemented)."""
        if mode == "human":
            position, velocity = self.state
            print(f"State: position={position:.3f}, velocity={velocity:.3f}")
        return None

    def suboptimal_expert(
        self,
        state: np.ndarray,
        K_position: float = 1.2176,
        K_velocity: float = 1.6073,
    ) -> np.ndarray:
        """Suboptimal expert policy for the double integrator environment.

        Args:
            state: Current state [position, velocity]
        Returns:
            action: Control input [acceleration]
        """

        position, velocity = state
        position_ref = self.target_position
        velocity_ref = 0.0

        K = np.array([[K_position, K_velocity]], dtype=np.float32)
        acceleration = -K @ np.array([[position - position_ref], [velocity - velocity_ref]])
        acceleration = np.clip(acceleration, -2.0, 2.0)

        return acceleration.astype(np.float32)


class DoubleIntegratorSuboptimalPolicy(NonTrainablePolicy):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__(observation_space, action_space)
        self.env = DoubleIntegratorEnv()
        self.K_position = 1.2176
        self.K_velocity = 1.6073

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return self.env.suboptimal_expert(obs, self.K_position, self.K_velocity)

    def set_K_values(self, K_position: float, K_velocity: float):
        self.K_position = K_position
        self.K_velocity = K_velocity