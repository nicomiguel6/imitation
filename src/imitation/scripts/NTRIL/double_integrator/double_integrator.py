"""Double Integrator Gymnasium Environment.

A double integrator is a classic control system where:
- State: [position, velocity]
- Action: [acceleration]
- Dynamics:
    position_ddot = acceleration

Observation (returned to the policy):
    [position, velocity, ref_position, ref_velocity]
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from imitation.policies.base import NonTrainablePolicy
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete


# ---------------------------------------------------------------------------
# Reference trajectory generator
# ---------------------------------------------------------------------------

def generate_reference_trajectory(
    T: int,
    dt: float = 1.0,
    mode: str = "constant",
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a reference trajectory for the double integrator.

    Args:
        T: Number of time steps (length of the trajectory).
        dt: Time step duration in seconds.
        mode: Trajectory type.  One of:

            ``"constant"``
                Fixed target position, zero velocity throughout.
                kwargs: ``target_position`` (float, default 0.0)

            ``"ramp"``
                Linearly interpolate position from *start_position* to
                *end_position* over T steps; velocity is the analytic
                derivative of the position profile (zero at the final step).
                kwargs: ``start_position`` (float, default 0.0),
                        ``end_position``   (float, required)

            ``"sinusoidal"``
                Sinusoidal position profile with analytically consistent
                velocity.
                kwargs: ``amplitude``  (float, default 1.0),
                        ``frequency``  (float Hz, default 0.1),
                        ``phase``      (float rad, default 0.0)

            ``"waypoints"``
                Piecewise-ramp through a fixed ordered list of positions.
                Each segment connects two consecutive waypoints with a
                constant-velocity ramp; velocity drops to zero at each
                junction.
                kwargs: ``positions``         (list[float], required, >= 2 elements),
                        ``steps_per_segment`` (int or list[int], optional;
                                              defaults to splitting T evenly)

            ``"random_waypoints"``
                Like ``"waypoints"`` but waypoints are drawn uniformly at
                random.
                kwargs: ``n_waypoints``  (int, default 4),
                        ``max_position`` (float, default 5.0)

        rng: Random number generator (used only for ``"random_waypoints"``).
        **kwargs: Mode-specific parameters described above.

    Returns:
        ref_traj: Array of shape ``(T, 2)``; each row is
                  ``[ref_position, ref_velocity]``.
    """
    if rng is None:
        rng = np.random.default_rng()

    ref_traj = np.zeros((T, 2), dtype=np.float32)

    if mode == "constant":
        target_position = float(kwargs.get("target_position", 0.0))
        ref_traj[:, 0] = target_position
        # ref_traj[:, 1] stays 0

    elif mode == "ramp":
        start = float(kwargs.get("start_position", 0.0))
        end = float(kwargs["end_position"])
        t = np.arange(T, dtype=np.float32)
        ref_traj[:, 0] = start + (end - start) * t / max(T - 1, 1)
        ramp_vel = (end - start) / (max(T - 1, 1) * dt)
        ref_traj[:-1, 1] = ramp_vel
        ref_traj[-1, 1] = 0.0  # hold zero at the final step

    elif mode == "sinusoidal":
        amplitude = float(kwargs.get("amplitude", 1.0))
        frequency = float(kwargs.get("frequency", 0.1))  # Hz
        phase = float(kwargs.get("phase", 0.0))          # rad
        t = np.arange(T, dtype=np.float32) * dt
        omega = 2.0 * np.pi * frequency
        ref_traj[:, 0] = amplitude * np.sin(omega * t + phase)
        ref_traj[:, 1] = amplitude * omega * np.cos(omega * t + phase)

    elif mode in ("waypoints", "random_waypoints"):
        if mode == "random_waypoints":
            n_waypoints = int(kwargs.get("n_waypoints", 4))
            max_pos = float(kwargs.get("max_position", 5.0))
            positions: List[float] = rng.uniform(-max_pos, max_pos, size=n_waypoints).tolist()
        else:
            positions = list(kwargs["positions"])

        n_segments = len(positions) - 1
        if n_segments < 1:
            raise ValueError("'waypoints' mode requires at least 2 positions.")

        sps = kwargs.get("steps_per_segment", None)
        if sps is None:
            base = T // n_segments
            remainder = T % n_segments
            steps_per_seg = [base + (1 if i < remainder else 0) for i in range(n_segments)]
        elif isinstance(sps, int):
            steps_per_seg = [sps] * n_segments
        else:
            steps_per_seg = list(sps)

        idx = 0
        for p_start, p_end, n_steps in zip(positions[:-1], positions[1:], steps_per_seg):
            if n_steps == 0 or idx >= T:
                continue
            end_idx = min(idx + n_steps, T)
            seg_len = end_idx - idx
            if seg_len == 1:
                ref_traj[idx, 0] = p_start
                ref_traj[idx, 1] = 0.0
            else:
                t_seg = np.arange(seg_len, dtype=np.float32)
                ref_traj[idx:end_idx, 0] = p_start + (p_end - p_start) * t_seg / (seg_len - 1)
                seg_vel = (p_end - p_start) / ((seg_len - 1) * dt)
                ref_traj[idx:end_idx - 1, 1] = seg_vel
                ref_traj[end_idx - 1, 1] = 0.0  # zero at each junction
            idx = end_idx

        # Hold the last waypoint if steps_per_seg didn't cover all T steps.
        if idx < T:
            ref_traj[idx:, 0] = positions[-1]
            ref_traj[idx:, 1] = 0.0

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: "
            "'constant', 'ramp', 'sinusoidal', 'waypoints', 'random_waypoints'."
        )

    return ref_traj


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DoubleIntegratorEnv(gym.Env):
    """Double integrator environment with reference-trajectory tracking.

    The agent controls a double integrator system to follow a reference
    trajectory.  The **observation** is the augmented state:

        ``[position, velocity, ref_position, ref_velocity]``

    where ``ref_position`` and ``ref_velocity`` are the desired values at the
    current time step, obtained from a reference trajectory supplied at
    ``reset()`` time.  If no trajectory is supplied, a constant reference at
    ``target_position`` with zero velocity is used (backward-compatible with
    the previous behaviour).

    Dynamics:
        x = [position, velocity]
        u = [acceleration]

        A = [[0, 1],
             [0, 0]]
        B = [[0],
             [1]]
    """

    def __init__(
        self,
        dt: float = 1.0,
        max_position: float = 50.0,
        max_velocity: float = 50.0,
        max_acceleration: float = 50.0,
        target_position: float = 0.0,
        position_tolerance: float = 0.1,
        velocity_tolerance: float = 0.1,
        position_cost_weight: float = 1.0,
        velocity_cost_weight: float = 1.0,
        control_cost_weight: float = 0.1,
        max_episode_seconds: float = 20.0,
        disturbance_magnitude: float = 0.0,
    ):
        """Initialize the double integrator environment.

        Args:
            dt: Time step for integration.
            max_position: Maximum absolute position (state bounds).
            max_velocity: Maximum absolute velocity (state bounds).
            max_acceleration: Maximum absolute acceleration (action bounds).
            target_position: Default reference position used when no reference
                trajectory is passed to ``reset()``.
            position_tolerance: Tolerance for reaching target position.
            velocity_tolerance: Tolerance for reaching target velocity.
            position_cost_weight: Weight for position error in reward.
            velocity_cost_weight: Weight for velocity error in reward.
            control_cost_weight: Weight for control effort in reward.
            max_episode_seconds: Physical duration of each episode in seconds.
                The number of steps is derived as
                ``int(round(max_episode_seconds / dt))`` so the episode length
                stays constant when ``dt`` changes.
            disturbance_magnitude: Scale of additive process disturbance.
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
        self.max_episode_seconds = max_episode_seconds
        self.max_episode_steps = int(round(max_episode_seconds / dt))
        self.disturbance_magnitude = disturbance_magnitude

        # Observation space: [position, velocity, ref_position, ref_velocity]
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

        # Action space: [acceleration]
        self.action_space = Box(
            low=np.array([-max_acceleration], dtype=np.float32),
            high=np.array([max_acceleration], dtype=np.float32),
        )

        # Internal state: [position, velocity]
        self.state: Optional[np.ndarray] = None
        self.step_count: int = 0

        # Continuous-time dynamics matrices (ẋ = A_c x + B_c u)
        self.A_c = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.B_c = np.array([[0.0], [1.0]])

        # Discrete-time matrices via ZOH — identical method to do-mpc's
        # LinearModel.discretize(), which calls scipy.signal.cont2discrete(method='zoh').
        # For the double integrator (A_c^2 = 0):
        #   A_d = [[1, dt], [0, 1]]
        #   B_d = [[dt^2/2], [dt]]
        n_x, n_u = self.A_c.shape[0], self.B_c.shape[1]
        A_d, B_d, _, _, _ = cont2discrete(
            (self.A_c, self.B_c, np.eye(n_x), np.zeros((n_x, n_u))),
            dt,
            method="zoh",
        )
        self.A_d = A_d
        self.B_d = B_d

        # Cost matrices (used for terminal cost via DARE).
        # DARE requires discrete-time A_d, B_d — not the continuous A_c, B_c.
        self.Q = np.diag([position_cost_weight, velocity_cost_weight])
        self.R = np.diag([control_cost_weight])
        self.P = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)

        # Reference trajectory: shape (T, 2), set at reset() time.
        self._ref_traj: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_reference(self) -> np.ndarray:
        """Return the reference state [ref_pos, ref_vel] for the current step."""
        t = min(self.step_count, len(self._ref_traj) - 1)
        return self._ref_traj[t]

    def _build_obs(self) -> np.ndarray:
        """Concatenate physical state and current reference into the observation."""
        return np.concatenate([self.state, self._current_reference()], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        state: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        Args:
            state: Optional initial physical state ``[position, velocity]``.
                   Sampled uniformly from ``[-5, 5]^2`` if not provided.
            seed: Random seed for reproducibility.
            options: Optional dictionary.  May contain:
                ``"reference_trajectory"``: array of shape ``(T, 2)`` with
                columns ``[ref_position, ref_velocity]``.  If absent, a
                constant reference at ``self.target_position`` is used.

        Returns:
            observation: Initial augmented observation
                         ``[position, velocity, ref_position, ref_velocity]``.
            info: Empty dictionary.
        """
        super().reset(seed=seed)

        if state is None:
            self.state = self.np_random.uniform(
                low=[-5.0, -5.0],
                high=[5.0, 5.0],
            ).astype(np.float32)
        else:
            self.state = state.astype(np.float32)

        self.step_count = 0

        # Set reference trajectory for this episode.
        ref_traj = None
        if options is not None:
            ref_traj = options.get("reference_trajectory", None)
        if ref_traj is None:
            ref_traj = generate_reference_trajectory(
                T=self.max_episode_steps,
                dt=self.dt,
                mode="constant",
                target_position=self.target_position,
            )
        self._ref_traj = np.asarray(ref_traj, dtype=np.float32)

        return self._build_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment forward one time step.

        Args:
            action: Acceleration control input ``[acceleration]``.

        Returns:
            observation: Next augmented observation
                         ``[position, velocity, ref_position, ref_velocity]``.
            reward: Negative tracking cost for this step.
            terminated: Always ``False`` (episodes end only via truncation).
            truncated: ``True`` when ``max_episode_steps`` is reached.
            info: Dictionary with position, velocity, and tracking errors.
        """
        action = np.asarray(action, dtype=np.float32).reshape(1,)
        action = np.clip(action, -self.max_acceleration, self.max_acceleration)

        # ZOH discrete step: x_{k+1} = A_d x_k + B_d u_k
        # Matches exactly the model do-mpc's LinearModel.discretize() produces,
        # so MPC predictions and environment trajectories share the same model.
        new_state = (
            self.A_d @ self.state.reshape(-1, 1) + self.B_d @ action.reshape(-1, 1)
        ).flatten()

        # Additive discrete-time disturbance w_k, ||w_k|| <= disturbance_magnitude.
        # Added directly to the state (not through ẋ*dt) so the magnitude is
        # consistent with how compute_approximate_linear_mrpi samples disturbances.
        if self.disturbance_magnitude > 0.0:
            direction = np.random.uniform(-1.0, 1.0, size=self.state.shape)
            norm = np.linalg.norm(direction)
            if norm > 0.0:
                direction = direction / norm
            new_state = new_state + direction * self.disturbance_magnitude

        self.state = new_state

        position = self.state[0]
        velocity = self.state[1]

        # Clamp to state bounds.
        if velocity > self.max_velocity:
            velocity = self.max_velocity
        if velocity < -self.max_velocity:
            velocity = -self.max_velocity
        if position > self.max_position:
            position = self.max_position
        if position < -self.max_position:
            position = -self.max_position
        # if position == -self.max_position and velocity < 0:
        #     velocity = 0
        # if position == self.max_position and velocity > 0:
        #     velocity = 0

        self.state = np.array([position, velocity], dtype=np.float32)
        self.step_count += 1

        # Reference at the current (post-step) time.
        ref = self._current_reference()
        ref_position = ref[0]
        ref_velocity = ref[1]

        position_error = position - ref_position
        velocity_error = velocity - ref_velocity
        control_effort = action[0] ** 2

        reward = -(
            self.position_cost_weight * position_error ** 2
            + self.velocity_cost_weight * velocity_error ** 2
            + self.control_cost_weight * control_effort
        )

        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        if truncated:
            # Terminal cost penalises the error relative to the final reference.
            final_ref = self._ref_traj[-1]
            error = self.state - final_ref
            terminal_cost = error.T @ self.P @ error
            reward -= terminal_cost.item()

        reward = np.float32(reward)
        self.state = self.state.astype(np.float32)

        info = {
            "position": position,
            "velocity": velocity,
            "ref_position": ref_position,
            "ref_velocity": ref_velocity,
            "position_error": position_error,
            "velocity_error": velocity_error,
            "target_reached": False,
        }

        return self._build_obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment (optional, not implemented)."""
        if mode == "human":
            position, velocity = self.state
            ref = self._current_reference()
            print(
                f"State: position={position:.3f}, velocity={velocity:.3f} | "
                f"Ref: position={ref[0]:.3f}, velocity={ref[1]:.3f}"
            )
        return None

    def suboptimal_expert(
        self,
        state: np.ndarray,
        reference_state: np.ndarray,
        K_position: float = 1.2176,
        K_velocity: float = 1.6073,
    ) -> np.ndarray:
        """Suboptimal feedback-gain policy for the double integrator.

        Computes ``u = -K * (x - x_ref)`` where ``K = [K_position, K_velocity]``.

        Args:
            state: Current physical state ``[position, velocity]``.
            reference_state: Reference state ``[ref_position, ref_velocity]``.
            K_position: Proportional gain on position error.
            K_velocity: Proportional gain on velocity error.

        Returns:
            action: Control input ``[acceleration]``.
        """
        K = np.array([[K_position, K_velocity]], dtype=np.float32)
        error = state[:2] - reference_state[:2]
        acceleration = -K @ error.reshape(2, 1)
        acceleration = np.clip(acceleration, -self.max_acceleration, self.max_acceleration)
        return acceleration.reshape(1,).astype(np.float32)


# ---------------------------------------------------------------------------
# NonTrainablePolicy wrapper
# ---------------------------------------------------------------------------

class DoubleIntegratorSuboptimalPolicy(NonTrainablePolicy):
    """Wraps the feedback-gain suboptimal expert as an SB3-compatible policy.

    The policy reads the augmented observation ``[pos, vel, ref_pos, ref_vel]``
    and delegates to :meth:`DoubleIntegratorEnv.suboptimal_expert`.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__(observation_space, action_space)
        self.env = DoubleIntegratorEnv()
        self.K_position = 1.2176
        self.K_velocity = 1.6073

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        # obs: [..., 4] — physical state in dims 0:2, reference in dims 2:4
        state = obs[..., :2]
        reference_state = obs[..., 2:]
        return self.env.suboptimal_expert(state, reference_state, self.K_position, self.K_velocity)

    def set_K_values(self, K_position: float, K_velocity: float):
        self.K_position = K_position
        self.K_velocity = K_velocity


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test the reference trajectory generator
    ref_traj = generate_reference_trajectory(T=200, dt=1.0, mode="sinusoidal", amplitude=1.0, frequency=0.1, phase=0.0)
    print(ref_traj)
    plt.plot(ref_traj[:, 0], label="Position")
    plt.plot(ref_traj[:, 1], label="Velocity")
    plt.legend()
    plt.show()