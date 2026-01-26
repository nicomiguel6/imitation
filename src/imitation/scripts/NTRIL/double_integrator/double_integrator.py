"""Double Integrator Gymnasium Environment.

A double integrator is a classic control system where:
- State: [x, x_dot, y, y_dot] (2D position and velocity)
- Action: [u_x, u_y] (2D acceleration/force)
- Dynamics:
    x_ddot = u_x
    y_ddot = u_y
"""

from typing import Dict, Optional, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class DoubleIntegratorEnv(gym.Env):
    """Double integrator environment for control tasks.

    The agent controls a double integrator system to reach a target position.
    The state consists of position and velocity, and the action is acceleration.

    Dynamics:
        x = [x, x_dot, y, y_dot]
        u = [u_x, u_y]

        x_ddot = u_x
        y_ddot = u_y

        A = [[0, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]
        B = [[0, 0],
             [1, 0],
             [0, 0],
             [0, 1]]

    where x is position, x_dot is velocity, y is position, y_dot is velocity, and u_x and u_y are the control inputs (acceleration).
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_position: float = 10.0,
        max_velocity: float = 10.0,
        max_acceleration: float = 50.0,
        target_x_position: float = 0.0,
        target_y_position: float = 0.0,
        x_position_tolerance: float = 0.1,
        y_position_tolerance: float = 0.1,
        x_velocity_tolerance: float = 0.1,
        y_velocity_tolerance: float = 0.1,
        x_position_cost_weight: float = 1.0,
        y_position_cost_weight: float = 1.0,
        x_velocity_cost_weight: float = 0.1,
        y_velocity_cost_weight: float = 0.1,
        control_cost_weight: float = 0.01,
        velocity_tolerance: float = 0.1,
        max_episode_steps: int = 200,
    ):
        """Initialize the double integrator environment.

        Args:
            dt: Time step for integration
            max_position: Maximum absolute position (bounds for initial state)
            max_velocity: Maximum absolute velocity (bounds for initial state)
            max_acceleration: Maximum absolute acceleration (action bounds)
            target_x_position: Target x position to reach
            target_y_position: Target y position to reach
            x_position_tolerance: Tolerance for reaching target x position
            y_position_tolerance: Tolerance for reaching target y position
            x_velocity_tolerance: Tolerance for reaching target x velocity
            y_velocity_tolerance: Tolerance for reaching target y velocity
            x_position_cost_weight: Weight for x position error in reward
            y_position_cost_weight: Weight for y position error in reward
            x_velocity_cost_weight: Weight for x velocity error in reward
            y_velocity_cost_weight: Weight for y velocity error in reward
            control_cost_weight: Weight for control effort in reward
            max_episode_steps: Maximum number of steps per episode
        """
        super().__init__()

        self.dt = dt
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.target_x_position = target_x_position
        self.target_y_position = target_y_position
        self.x_position_tolerance = x_position_tolerance
        self.y_position_tolerance = y_position_tolerance
        self.x_velocity_tolerance = x_velocity_tolerance
        self.y_velocity_tolerance = y_velocity_tolerance
        self.x_position_cost_weight = x_position_cost_weight
        self.y_position_cost_weight = y_position_cost_weight
        self.x_velocity_cost_weight = x_velocity_cost_weight
        self.y_velocity_cost_weight = y_velocity_cost_weight
        self.control_cost_weight = control_cost_weight
        self.max_episode_steps = max_episode_steps

        # Observation space: [x, x_dot, y, y_dot]
        self.observation_space = Box(
            low=np.array(
                [-max_position, -max_velocity, -max_position, -max_velocity],
                dtype=np.float32,
            ),
            high=np.array(
                [max_position, max_velocity, max_position, max_velocity],
                dtype=np.float32,
            ),
        )

        # Action space: [u_x, u_y]
        self.action_space = Box(
            low=np.array([-max_acceleration, -max_acceleration], dtype=np.float32),
            high=np.array([max_acceleration, max_acceleration], dtype=np.float32),
        )

        # State: [x, x_dot, y, y_dot]
        self.state = None
        self.step_count = 0

        # Dynamics
        self.A = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        self.B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional dictionary with reset options

        Returns:
            observation: Initial observation [x, x_dot, y, y_dot]
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        # Sample initial state uniformly from bounds
        self.state = self.np_random.uniform(
            low=[
                -self.max_position,
                -self.max_velocity,
                -self.max_position,
                -self.max_velocity,
            ],
            high=[
                self.max_position,
                self.max_velocity,
                self.max_position,
                self.max_velocity,
            ],
        ).astype(np.float32)

        self.step_count = 0

        # # Make initial velocity 0
        # self.state[1] = 0.1
        # self.state[3] = 0.1

        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment forward one time step.

        Args:
            action: Acceleration control input [u_x, u_y]

        Returns:
            observation: Next observation [x, x_dot, y, y_dot]
            reward: Reward for this step
            terminated: Whether episode terminated (reached target)
            truncated: Whether episode was truncated (max steps)
            info: Dictionary with additional information
        """
        # Clip action to valid range
        action = np.clip(action, -self.max_acceleration, self.max_acceleration)

        # Update state using Euler integration
        xdot = self.A @ self.state + self.B @ action
        self.state = self.state + xdot * self.dt

        # Clip state to bounds
        self.state[0] = np.clip(self.state[0], -self.max_position, self.max_position)
        self.state[1] = np.clip(self.state[1], -self.max_velocity, self.max_velocity)
        self.state[2] = np.clip(self.state[2], -self.max_position, self.max_position)
        self.state[3] = np.clip(self.state[3], -self.max_velocity, self.max_velocity)

        self.step_count += 1

        # Extract state components after update: [x, x_dot, y, y_dot]
        x, x_dot, y, y_dot = self.state

        # Compute reward (negative cost)
        # Target is at (target_x_position, target_y_position) with zero velocity
        x_position_error = x - self.target_x_position
        y_position_error = y - self.target_y_position
        x_velocity_error = x_dot  # Target velocity is 0
        y_velocity_error = y_dot  # Target velocity is 0

        # Position and velocity errors squared
        x_position_error_sq = x_position_error**2
        y_position_error_sq = y_position_error**2
        x_velocity_error_sq = x_velocity_error**2
        y_velocity_error_sq = y_velocity_error**2
        # Control effort
        control_effort = action[0] ** 2 + action[1] ** 2

        reward = -(
            self.x_position_cost_weight * x_position_error_sq
            + self.y_position_cost_weight * y_position_error_sq
            + self.x_velocity_cost_weight * x_velocity_error_sq
            + self.y_velocity_cost_weight * y_velocity_error_sq
            + self.control_cost_weight * control_effort
        )

        # Check if target reached
        x_position_reached = abs(x_position_error) < self.x_position_tolerance
        y_position_reached = abs(y_position_error) < self.y_position_tolerance
        x_velocity_reached = abs(x_velocity_error) < self.x_velocity_tolerance
        y_velocity_reached = abs(y_velocity_error) < self.y_velocity_tolerance
        terminated = (
            x_position_reached
            and y_position_reached
            and x_velocity_reached
            and y_velocity_reached
        )

        # Check if max steps reached
        truncated = self.step_count >= self.max_episode_steps

        # Ensure returned state is float32
        self.state = self.state.astype(np.float32)

        info = {
            "x": x,
            "x_dot": x_dot,
            "y": y,
            "y_dot": y_dot,
            "x_position_error": x_position_error,
            "y_position_error": y_position_error,
            "x_velocity_error": x_velocity_error,
            "y_velocity_error": y_velocity_error,
            "target_reached": terminated,
        }

        return self.state.copy(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment (optional, not implemented)."""
        if mode == "human":
            x, x_dot, y, y_dot = self.state
            print(f"State: x={x:.3f}, x_dot={x_dot:.3f}, y={y:.3f}, y_dot={y_dot:.3f}")
        return None

    def suboptimal_expert(
        self,
        state: np.ndarray,
        K_x: float = 5.0,
        K_xdot: float = 2.0,
        K_y: float = 5.0,
        K_ydot: float = 2.0,
    ) -> np.ndarray:
        """Suboptimal expert policy for the double integrator environment.

        Args:
            state: Current state [x, x_dot, y, y_dot]
            reference_state: Reference state [x_ref, x_dot_ref, y_ref, y_dot_ref]
        Returns:
            action: Control input [u_x, u_y]
        """

        x, x_dot, y, y_dot = state
        x_ref = self.target_x_position
        x_dot_ref = 0.0
        y_ref = self.target_y_position
        y_dot_ref = 0.0

        u_x = K_x * (x_ref - x) + K_xdot * (x_dot_ref - x_dot)
        u_y = K_y * (y_ref - y) + K_ydot * (y_dot_ref - y_dot)

        return np.array([u_x, u_y])



