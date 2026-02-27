"""Robust Tube Model Predictive Control for data augmentation in NTRIL pipeline."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import gymnasium as gym
import numpy as np
import torch as th
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are, solve_continuous_are
import casadi as ca
import do_mpc
import cdd
import pytope

from imitation.data import types
from imitation.policies.base import NonTrainablePolicy
from imitation.util import util
from imitation.data import types


class RobustTubeMPC:
    """Robust Tube Model Predictive Control for trajectory augmentation.

    This class implements linear robust tube MPC under a specified disturbance set
    to augment state-action pairs using Tagliabue's method.
    """

    def __init__(
        self,
        horizon: int = 10,
        episode_length: int = 200,
        time_step: float = 0.1,
        disturbance_bound: Optional[float] = None,
        tube_radius: Optional[float] = None,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        disturbance_vertices: Optional[np.ndarray] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linearization_method: str = "finite_difference",
        finite_diff_eps: float = 1e-6,
        initial_state: Optional[np.ndarray] = None,
        reference_trajectory: Optional[types.Trajectory] = None,
    ):
        """Initialize Robust Tube MPC.

        Args:
            horizon: MPC prediction horizon
            episode_length: Length of episode
            time_step: discretization time step for MPC
            disturbance_bound: Bound for disturbance set
            tube_radius: Radius of the robust tube
            A: State Dynamics matrix
            B: Control Dynamics matrix
            Q: State cost matrix (if None, will be identity)
            R: Control cost matrix (if None, will be identity)
            disturbance_vertices: Vertices of disturbance distribution
            state_bounds: Tuple of (lower_bound, upper_bound) for states. If None, assumes no constraints.
            control_bounds: Tuple of (lower_bound, upper_bound) for controls. If None, assumes no constraints.
            linearization_method: Method for linearizing dynamics
            finite_diff_eps: Epsilon for finite difference linearization
            reference_trajectory: Reference trajectory to follow.
        """
        self.horizon = horizon
        self.episode_length = episode_length
        self.time_step = time_step

        # Handle defaults for disturbance_bound and tube_radius if not given
        self.disturbance_bound = disturbance_bound
        self.tube_radius = tube_radius

        # Ensure that A and B matrices are provided
        if A is None or B is None:
            raise ValueError("System dynamic matrices A and B must be provided.")
        self.A = A
        self.B = B

        # Provide defaults for Q and R if they are not supplied
        state_dim = np.shape(A)[1]
        action_dim = np.shape(B)[1]
        self.Q = Q if Q is not None else np.eye(state_dim)
        self.R = R if R is not None else np.eye(action_dim)

        self.disturbance_vertices = disturbance_vertices

        # Handle state and control bounds: if not provided, allow unconstrained, i.e., set to None
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds

        self.linearization_method = linearization_method
        self.finite_diff_eps = finite_diff_eps

        # Default initial state to zero if not given
        self.initial_state = (
            initial_state if initial_state is not None else np.zeros(state_dim)
        )

        self.state_dim = np.shape(A)[1]
        self.action_dim = np.shape(B)[1]
        self.xs = []
        self.us = []

        if reference_trajectory is None: # default to all zeros if none is provided, this changes the cost function to a minimization problem instead of tracking problem
            self.reference_trajectory = types.Trajectory(obs=np.zeros((self.episode_length + 1, self.state_dim)), acts=np.zeros((self.episode_length, self.action_dim)), infos=np.array([{}] * self.episode_length), terminal=True)
        else:
            self.reference_trajectory = reference_trajectory
    
    def setup(self):
        """Initialize necessary objects for setting up MPC"""

        # ------------ UNDISTURBED MODEL SETUP ------------- #
        # for use in generating nominal control
        model_type = "continuous"
        nominal_model = do_mpc.model.LinearModel(model_type)

        _x = nominal_model.set_variable(
            var_type="_x", var_name="x", shape=(self.state_dim, 1)
        )
        _u = nominal_model.set_variable(
            var_type="_u", var_name="u", shape=(self.action_dim, 1)
        )

        nominal_x_next = self.A @ _x + self.B @ _u

        nominal_model.set_rhs("x", nominal_x_next)

        # ------------ REFERENCE TRAJECTORY SETUP ------------- #
        # Shape is (state_dim, 1): a single reference per prediction step.
        # tvp_fun assigns a different value for each k in [0, n_horizon].
        state_set_point = nominal_model.set_variable(var_type="_tvp", var_name="state_set_point", shape=(self.state_dim, 1))
        control_set_point = nominal_model.set_variable(var_type="_tvp", var_name="control_set_point", shape=(self.action_dim, 1))

        nominal_model.setup()

        # ------------ PRELIMINARY SETS ------------- #

        # convert A and B to discrete time
        disc_model = nominal_model.discretize(t_step=self.time_step)
        self.A_d = disc_model._A
        self.B_d = disc_model._B
        # self.A_d = self.A
        # self.B_d = self.B

        # Solve for P, K matrices
        self.P = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)
        self.K = -np.linalg.inv(self.B_d.T @ self.P @ self.B_d + self.R) @ (
            self.B_d.T @ self.P @ self.A_d
        )

        if self.disturbance_bound is not None:

            # Closed loop
            self.Ak = self.A_d + self.B_d @ self.K

            # Set up disturbance polytope
            self.disturbance_polytope = pytope.Polytope(self.disturbance_vertices)

            # Compute mrpi set
            self.A_F, self.b_F = compute_mrpi_hrep(
                self.Ak, self.disturbance_polytope.A, self.disturbance_polytope.b
            )
            self.Z = pytope.Polytope(self.A_F, self.b_F)

            # Tighten state and constraint bounds accordingly
            if self.state_bounds is not None:
                A_x_t, b_x_t = tighten_state_constraints(
                    self.state_bounds, self.A_F, self.b_F
                )
                self.A_x_t = A_x_t
                self.b_x_t = b_x_t

            if self.control_bounds is not None:
                A_u_t, b_u_t = tighten_control_constraints(
                    self.control_bounds, self.K, self.A_F, self.b_F
                )
                self.A_u_t = A_u_t
                self.b_u_t = b_u_t
        else:
                self.A_x, self.b_x = box_to_Ab(self.state_bounds[0], self.state_bounds[1])
                self.A_u, self.b_u = box_to_Ab(self.control_bounds[0], self.control_bounds[1])


        # ------------ CONTROLLER SETUP ------------- #
        # solver for nominal control
        mpc = do_mpc.controller.MPC(nominal_model)
        setup_mpc = {
            "n_horizon": self.horizon,
            "t_step": self.time_step,
            "state_discretization": "collocation",
            "collocation_type": "radau",
            "collocation_deg": 3,
            "collocation_ni": 1,
            "store_full_solution": True,
        }
        # setup_mpc = {
        #     'n_horizon': self.horizon,
        #     't_step': self.time_step,
        #     'state_discretization': 'discrete',
        #     'store_full_solution': True,
        # }
        mpc.set_param(**setup_mpc)


        # mpc.set_param(**setup_mpc)

        # Objective
        x = nominal_model.x["x"]
        u = nominal_model.u["u"]

        lterm = (x-nominal_model.tvp["state_set_point"]).T @ self.Q @ (x-nominal_model.tvp["state_set_point"]) + (u-nominal_model.tvp["control_set_point"]).T @ self.R @ (u-nominal_model.tvp["control_set_point"])
        mterm = (x-nominal_model.tvp["state_set_point"]).T @ self.P @ (x-nominal_model.tvp["state_set_point"]) # terminal cost

        if self.disturbance_bound is not None:
            # lower bounds of the states
            mpc.bounds["lower", "_x", "x"] = -np.array([self.b_x_t[1], self.b_x_t[3]])

            # upper bounds of the states
            mpc.bounds["upper", "_x", "x"] = np.array([self.b_x_t[0], self.b_x_t[2]])

            # lower bounds of the input
            mpc.bounds["lower", "_u", "u"] = -np.array([self.b_u_t[0]])

            # upper bounds of the input
            mpc.bounds["upper", "_u", "u"] = np.array([self.b_u_t[1]])
        else:
            # lower bounds of the states
            mpc.bounds["lower", "_x", "x"] = -np.array([self.b_x[1], self.b_x[3]])

            # upper bounds of the states
            mpc.bounds["upper", "_x", "x"] = np.array([self.b_x[0], self.b_x[2]])

            # lower bounds of the input
            mpc.bounds["lower", "_u", "u"] = -np.array([self.b_u[0]])

            # upper bounds of the input
            mpc.bounds["upper", "_u", "u"] = np.array([self.b_u[1]])

        mpc.settings.set_linear_solver("ma57")
        mpc.settings.supress_ipopt_output()
        mpc.set_objective(mterm=mterm, lterm=lterm)
        #mpc.set_rterm(ca.SX(self.R))

        # ------------ TIME VARYING PARAMETERS (TVP) SETUP ------------- #
        # set up time varying parameters (tvp)
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            t_idx = int(t_now / self.time_step)
            n_obs = len(self.reference_trajectory.obs)
            n_acts = len(self.reference_trajectory.acts)
            for k in range(self.horizon + 1):
                obs_idx = min(t_idx + k, n_obs - 1)
                act_idx = min(t_idx + k, n_acts - 1)
                tvp_template["_tvp", k, "state_set_point"] = self.reference_trajectory.obs[obs_idx].reshape(-1, 1)
                tvp_template["_tvp", k, "control_set_point"] = self.reference_trajectory.acts[act_idx].reshape(-1, 1)
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()

        self.mpc = mpc


        if self.disturbance_bound is not None:

            # ------------ DISTURBED MODEL & SIMULATOR SETUP ------------- #
            disturbed_model = do_mpc.model.Model(model_type)

            _xd = disturbed_model.set_variable(
                var_type="_x", var_name="xd", shape=(self.state_dim, 1)
            )
            _ud = disturbed_model.set_variable(
                var_type="_u", var_name="ud", shape=(self.action_dim, 1)
            )

            # Set disturbance
            _d = disturbed_model.set_variable(
                var_type="_p", var_name="d", shape=(self.state_dim, 1)
            )

            disturbed_x_next = self.A @ _xd + self.B @ _ud + _d
            disturbed_model.set_rhs("xd", disturbed_x_next)
            disturbed_model.setup()

            self.simulator = do_mpc.simulator.Simulator(disturbed_model)

            d_template = self.simulator.get_p_template()

            def d_fun(t_now):
                d_template["d"] = sample_from_disturbance(self.disturbance_polytope)
                return d_template

            self.simulator.set_p_fun(d_fun)
        
        else:
            self.simulator = do_mpc.simulator.Simulator(nominal_model)

            sim_tvp_template = self.simulator.get_tvp_template()

            def sim_tvp_fun(t_now):
                t_idx = int(t_now / self.time_step)
                n_obs = len(self.reference_trajectory.obs)
                n_acts = len(self.reference_trajectory.acts)
                obs_idx = min(t_idx, n_obs - 1)
                act_idx = min(t_idx, n_acts - 1)
                sim_tvp_template["state_set_point"] = self.reference_trajectory.obs[obs_idx].reshape(-1, 1)
                sim_tvp_template["control_set_point"] = self.reference_trajectory.acts[act_idx].reshape(-1, 1)
                return sim_tvp_template

            self.simulator.set_tvp_fun(sim_tvp_fun)

        self.simulator.set_param(t_step=self.time_step)
        self.simulator.setup()

        print(
            "Nominal and disturbed models, controller, and simulator setup completed!"
        )

    def set_reference_trajectory(self, reference_trajectory: types.Trajectory) -> None:
        """Update the reference trajectory without rebuilding the MPC problem.

        The compiled NLP, MRPI sets, and all CasADi machinery are reused.
        Only the data read by tvp_fun at each solve step changes.

        Args:
            reference_trajectory: New reference trajectory to track.
        """
        self.reference_trajectory = reference_trajectory
        # Reset internal time so t_idx starts at 0 for the new trajectory.
        if self.mpc is not None:
            self.mpc.reset_history()
            self.mpc.t0 = 0.0
        if self.simulator is not None:
            self.simulator.reset_history()
            self.simulator.t0 = 0.0
        self.xs = []
        self.us = []

    def solve_mpc(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve one MPC step given the current state.

        Args:
            state (np.ndarray): Current state of the system.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (next_state, applied_control).
        """
        if self.mpc is None or self.simulator is None:
            raise ValueError(
                "Must set MPC and Simulator objects before attempting to solve"
            )

        # Set initial state for MPC and simulator
        self.mpc.x0 = state
        self.simulator.x0 = state
        self.mpc.set_initial_guess()

        # Compute nominal optimal control from MPC
        u_nom = self.mpc.make_step(state)
        
        # Predict nominal state (first predicted state in the horizon)
        nominal_state = self.mpc.data.prediction(("_x", "x"))[:, 0]

        # Convert to column vectors for disturbance rejection control law
        state_col = np.asarray(state).reshape(-1, 1)
        nominal_col = np.asarray(nominal_state).reshape(-1, 1)

        if self.disturbance_bound is not None:
            applied_u = u_nom + self.K @ (state_col - nominal_col)
        else:
            applied_u = u_nom

        # Simulate the next state using the simulator
        next_state = self.simulator.make_step(u0=applied_u)

        self.xs.append(state)
        self.us.append(applied_u)

        return next_state, applied_u.flatten()

    def augment_trajectory(
        self,
        trajectory: types.Trajectory,
        partial_horizon: int = 20,
        k_timesteps: int = 5,
        n_augmentations: int = 5,
    ) -> Sequence[types.TrajectoryWithRew]:
        """Augment a trajectory using robust tube MPC. Sample points every k_timesteps and propagate partial trajectories following ancillary controller u = u0 + K(x-x0) .

        Args:
            trajectory: Trajectory to augment
            partial_horizon: Length of partial trajectory
            k_timesteps: Interval of time steps to select sample from
            n_augmentations: Number of augmented samples per transition

        Returns:
            Collection of partial trajectories sampled from RTMPC trajectory
        """

        augmented_infos = []

        augmented_trajs = []

        for t in range(len(trajectory.obs) - 1):
            if t % k_timesteps == 0 and t + partial_horizon < len(trajectory.obs):

                augmented_trajs_t = []

                # Extract current nominal state and action
                current_nominal_state = trajectory.obs[t]
                current_nominal_action = trajectory.acts[t]
                current_noise_level = trajectory.infos[t].get("noise_level")
                total_applied_noise_sum = trajectory.infos[-1].get("total_applied_noise_sum")

                # Sample points at center of bounding box facets
                tube_set = current_nominal_state + self.Z
                approx_tube = get_approximate_tube(tube_set)
                samples = get_samples(approx_tube, corners=False)

                # simulate partial trajectory for each samples (should be cheap as it's only propagating dynamics)
                for sample in samples:

                    # initialize trajectory builder
                    builder = util.TrajectoryBuilder()
                    builder.start_episode(initial_obs=sample)

                    # initialize state and action
                    x = sample
                    u = current_nominal_action
                    self.simulator.x0 = x

                    # propagate dynamics
                    for t_step in range(partial_horizon - 1):
                        u_applied = current_nominal_action + self.K @ (
                            x.T.reshape(
                                -1, 1
                            )
                            - current_nominal_state.reshape(
                                -1, 1
                            )
                        )
                        x_next = self.simulator.make_step(u0=u_applied)
                        # calculate tracking cost
                        tracking_cost = self._compute_tracking_cost_metric(
                            trajectory.obs[t + t_step], x, trajectory.acts[t + t_step], u_applied, t + t_step
                        )
                        # calculate margin to safety violation
                        margin_safety = self._compute_margin_safety_violation(x)
                        builder.add_step(
                            action=u_applied.flatten(),
                            next_obs=x_next.flatten(),
                            reward=0.0,
                            info={
                                "tracking_cost": tracking_cost,
                                "margin_safety": margin_safety,
                                "noise_level": current_noise_level,
                                "total_applied_noise_sum": total_applied_noise_sum,
                            },
                        )
                        x = x_next

                        current_nominal_state = trajectory.obs[t + t_step]
                        current_nominal_action = trajectory.acts[t + t_step]

                    # finalize trajectory
                    traj = builder.finish(terminal=True)

                    augmented_trajs_t.append(traj)

                    augmented_infos.append(
                        {
                            "original_timestep": t,
                            "augmentation_method": "robust_tube_mpc",
                        }
                    )

                augmented_trajs.extend(augmented_trajs_t)

        return augmented_trajs

    def _sample_disturbance(self, scale: float = 1.0) -> np.ndarray:
        """Sample a disturbance from the disturbance set.

        Args:
            scale: Scaling factor for disturbance magnitude

        Returns:
            Sampled disturbance vector
        """
        if self.state_dim is None:
            raise ValueError("State dimension not set")

        # Sample from uniform ball with radius = disturbance_bound * scale
        direction = np.random.randn(self.state_dim)
        direction /= np.linalg.norm(direction)
        radius = np.random.uniform(0, self.disturbance_bound * scale)
        return direction * radius

    def _compute_tracking_cost_metric(
        self, nominal_state: np.ndarray, x: np.ndarray, nominal_action: Optional[np.ndarray], u: Optional[np.ndarray], k: int
    ) -> np.float64:
        """Compute tracking cost metric for individual data point

        Args:
            nominal_state (np.ndarray): nominal state
            nominal_action (Optional[np.ndarray]): nominal action
            x (np.ndarray): current state
            u (Optional[np.ndarray]): current input (if not terminal state)
            k (int): timestep of interest

        Returns:
            np.float64: tracking cost J from MPC
        """
        # extract reference solution
        x_difference = x - nominal_state
        if u:
            u_difference = u - nominal_action
            return (
                x_difference.T @ self.Q @ x_difference
                + u_difference.T @ self.R @ u_difference
            )

        return x_difference.T @ self.Q @ x_difference

    def _compute_margin_safety_violation(self, x: np.ndarray) -> np.float64:
        """Calculate margin to safety violation of a given state relative to tightened state constraints

        Args:
            x (np.ndarray): state

        Returns:
            np.float64: margin for that state (higher is better)
        """
        if self.state_bounds is None:
            return 1e6

        # calculate slack for each timestep
        min_margin = float("inf")
        for a_row, b_val in zip(self.A_x_t, self.b_x_t):
            margin = b_val - a_row.T @ x
            if margin < min_margin:
                min_margin = margin

        return min_margin


class RobustTubeMPCPolicy(NonTrainablePolicy):
    """Policy wrapper that exposes RobustTubeMPC as a BasePolicy."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        auto_setup: bool = True,
        **mpc_kwargs: Any,
    ):
        """Builds a BasePolicy-compatible wrapper around RobustTubeMPC."""
        super().__init__(observation_space=observation_space, action_space=action_space)
        self._auto_setup = auto_setup
        self._mpc_kwargs = mpc_kwargs
        self._controllers: List[RobustTubeMPC] = []

    def _build_controller(self) -> RobustTubeMPC:
        controller = RobustTubeMPC(**self._mpc_kwargs)
        if self._auto_setup:
            controller.setup()
        return controller

    def _ensure_controllers(self, n_envs: int) -> None:
        if not self._controllers or len(self._controllers) != n_envs:
            self._controllers = [self._build_controller() for _ in range(n_envs)]

    def reset(
        self,
        *,
        env_index: Optional[int] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        """Reset controller state for one or all environments."""
        if not self._controllers:
            return
        if env_index is None:
            controllers = self._controllers
        else:
            controllers = [self._controllers[env_index]]

        for controller in controllers:
            controller.xs = []
            controller.us = []
            if initial_state is not None:
                controller.initial_state = np.asarray(initial_state)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, gym.spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def _maybe_batch_obs(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray], th.Tensor, Dict[str, th.Tensor]]
    ) -> Union[np.ndarray, types.DictObs]:
        if isinstance(observation, dict):
            if not isinstance(self.observation_space, gym.spaces.Dict):
                raise ValueError("Dictionary observation provided for non-dict space.")
            batched: Dict[str, np.ndarray] = {}
            for key, value in observation.items():
                if isinstance(value, th.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = np.asarray(value)
                space_shape = self.observation_space.spaces[key].shape
                if arr.shape == space_shape:
                    arr = arr[None, ...]
                batched[key] = arr
            return types.DictObs(batched)

        if isinstance(observation, th.Tensor):
            arr = observation.detach().cpu().numpy()
        else:
            arr = np.asarray(observation)
        if arr.shape == self.observation_space.shape:
            arr = arr[None, ...]
        return arr

    def _choose_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        self._ensure_controllers(1)
        obs_array = types.maybe_unwrap_dictobs(obs)
        _, action = self._controllers[0].solve_mpc(np.asarray(obs_array))
        return self._clip_action(action)

    def predict(  # type: ignore[override]
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray], th.Tensor, Dict[str, th.Tensor]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        batched_obs = self._maybe_batch_obs(observation)
        n_envs = len(batched_obs)
        self._ensure_controllers(n_envs)

        if episode_start is not None:
            episode_start = np.asarray(episode_start, dtype=bool)
            if episode_start.shape[0] != n_envs:
                raise ValueError(
                    f"episode_start has length {episode_start.shape[0]} but expected {n_envs}",
                )

        actions: List[np.ndarray] = []
        for env_idx, obs in enumerate(batched_obs):
            obs_unwrapped = types.maybe_unwrap_dictobs(obs)
            if episode_start is not None and episode_start[env_idx]:
                self.reset(env_index=env_idx, initial_state=obs_unwrapped)
            _, action = self._controllers[env_idx].solve_mpc(np.asarray(obs_unwrapped))
            action = self._clip_action(action)
            if not self.action_space.contains(action):
                raise ValueError(
                    f"Action {action} is not in action space {self.action_space}",
                )
            actions.append(action)

        return np.stack(actions, axis=0), state


def tighten_state_constraints(
    state_constraint_vertices: List[np.ndarray], A_F: np.ndarray, b_F: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Tightens state constraints using calculated mRPI H-rep

    Args:
        state_constraint_vertices (np.ndarray): list of state constraints of the form: [lower, upper]
        A_F (np.ndarray): A-matrix of mRPI H-rep set
        b_F (np.ndarray): b-vector of mRPI H-rep set

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing tightened A-matrix and tightened b-vector of state constraints
    """
    A_x, b_x = box_to_Ab(state_constraint_vertices[0], state_constraint_vertices[1])

    A_x_t, b_x_t = minkowski_difference_Hrep(A_x, b_x, A_F, b_F)

    return A_x_t, b_x_t


def tighten_control_constraints(
    control_constraint_vertices: np.ndarray,
    K: np.ndarray,
    A_F: np.ndarray,
    b_F: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tightens control constraints using calculated mRPI H-rep

    Args:
        state_constraint_vertices (np.ndarray): list of control constraints of the form: [lower, upper]
        A_F (np.ndarray): A-matrix of mRPI H-rep set
        b_F (np.ndarray): b-vector of mRPI H-rep set

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing tightened A-matrix and tightened b-vector of control constraints
    """
    A_u, b_u = box_to_Ab(control_constraint_vertices[0], control_constraint_vertices[1])
    A_u_t, b_u_t = tighten_by_linear_image_Hrep(A_u, b_u, K, A_F, b_F)

    return A_u_t, b_u_t


def get_approximate_tube(Z_polyhedron: pytope.Polytope) -> pytope.Polytope:
    """Generate outer approximation bounding box for tube cross section Z

    Args:
        Z_polyhedron (pytope.Polytope): Tube polyhedron

    Returns:
        pytope.Polytope: Outer approximation polyhedron
    """
    # Extract vertices of polyhedron
    try:
        verts = Z_polyhedron.V
    except RuntimeError as e:
        # cdd numerical inconsistency — inspect Z_polyhedron here.
        # Useful attributes:
        #   Z_polyhedron.A, Z_polyhedron.b  — H-rep (Ax <= b)
        #   np.linalg.cond(Z_polyhedron.A)  — condition number of A
        print(f"[DEBUG] cdd RuntimeError: {e}")
        print(f"[DEBUG] H-rep A:\n{Z_polyhedron.A}")
        print(f"[DEBUG] H-rep b:\n{Z_polyhedron.b}")
        breakpoint()
        raise

    max_points = []
    min_points = []
    b_matrix = []
    A_matrix_list = []

    # Iterate over each dim to extract max and min value, add to b matrix
    n_dim = len(verts[0])
    for dim in range(n_dim):
        temp = verts[:, dim]
        tmp_vector = np.zeros((2, n_dim))
        tmp_vector[:, dim] = np.array([1, -1])
        max_points.append(np.max(temp))
        min_points.append(np.min(temp))
        b_matrix.append(np.max(temp))
        b_matrix.append(-np.min(temp))
        A_matrix_list.append(tmp_vector)

    # form A_matrix
    A_matrix = np.vstack(A_matrix_list)

    return pytope.Polytope(A_matrix, b_matrix)


def get_samples(
    Z_polyhedron: pytope.Polytope, corners: bool = True
) -> List[np.ndarray]:
    """Get corner or face-center samples from polyhedron

    Args:
        Z_polyhedron (pytope.Polytope): polyhedron to sample
        corners (bool, optional): Get corner samples. Defaults to True.

    Returns:
        list[np.ndarray]: list of sample points
    """
    n_dim = len(Z_polyhedron.V[0])
    if corners:
        # Collect corner cases
        samples = Z_polyhedron.V
    else:
        # Collect facet-centers

        samples = []
        midpoints = []
        minpoints = []
        maxpoints = []

        # Iterate over each dimension, extract midpoints
        for dim in range(n_dim):
            tmp = Z_polyhedron.V[:, dim]
            midpoints.append((np.max(tmp) + np.min(tmp)) / 2.0)
            minpoints.append(np.min(tmp))
            maxpoints.append(np.max(tmp))

        # Form vertices
        for i_dim in range(n_dim):
            for j_dim in range(n_dim):
                if j_dim == i_dim:
                    continue

                samples.append(np.array([midpoints[i_dim], minpoints[j_dim]]))
                samples.append(np.array([midpoints[i_dim], maxpoints[j_dim]]))

    return samples


def get_vertices(A, b):
    """Converts H-rep into a list of vertices"""

    # Form h-matrix and extract generators
    mat = cdd.Matrix(np.hstack([b.reshape(-1, 1), -A]), number_type="float")
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    generators = poly.get_generators()

    # Convert to list of vertices
    vertices = []
    for generator in generators:
        if generator[0] == 1:
            vertices.append(generator[1:])

    return np.array(vertices, dtype=float)


def sample_from_disturbance(W_polyhedron, n_samples=1):
    """Sample uniformly from the disturbance polytope."""
    vertices = W_polyhedron.V

    # Method 1: Rejection sampling for small polytopes
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    samples = []
    while len(samples) < n_samples:
        candidate = np.random.uniform(min_coords, max_coords)
        if W_polyhedron.contains(candidate):
            samples.append(candidate)

    return np.array(samples) if n_samples > 1 else samples[0]


def support_function(A, b, d):
    # cdd wants [b | A] with inequalities in form b + A x >= 0
    V = get_vertices(A, b)

    d = np.asarray(d)
    d = np.atleast_2d(d)  # shape (m, n_dirs)
    return np.max(V @ d, axis=0)


def minkowski_difference_Hrep(A_x, b_x, A_F, b_F):
    """
    Tighten state constraints X ⊖ F in H-rep.
    A_x, b_x : original state constraints
    A_F, b_F : H-rep of disturbance/error set F
    """
    import numpy as np

    b_tight = []
    for a, b in zip(A_x, b_x):
        h = support_function_lp(A_F, b_F, a)  # your routine
        b_tight.append(b - h)
    return np.array(A_x, dtype=float), np.array(b_tight, dtype=float)


def tighten_by_linear_image_Hrep(A_u, b_u, K, A_F, b_F):
    """
    Tighten input constraints U ⊖ K F in H-rep.
    A_u, b_u : original input constraints
    K        : feedback gain matrix
    A_F, b_F : H-rep of disturbance/error set F
    """
    b_tight = []
    for a, b in zip(A_u, b_u):
        d = K.T @ a.reshape(-1, 1)  # map direction
        h = support_function_lp(A_F, b_F, d.flatten())
        b_tight.append(b - h)
    return np.array(A_u, dtype=float), np.array(b_tight, dtype=float)


def box_to_Ab(lower, upper):
    """
    Convert simple box constraints (lower <= x <= upper)
    into H-representation A, b such that A x <= b.
    """
    lower = np.array(lower).flatten()
    upper = np.array(upper).flatten()
    n = len(lower)

    # For each dimension i:
    #  x_i <= upper_i      ->  A row = e_i,   b = upper_i
    # -x_i <= -lower_i     ->  A row = -e_i,  b = -lower_i
    A = []
    b = []
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A.append(e_i)
        b.append(upper[i])
        A.append(-e_i)
        b.append(-lower[i])

    return np.vstack(A), np.array(b)


def support_function_lp(A, b, d):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    h_repr = np.hstack([b.reshape(-1, 1)])

    lp_mat = cdd.Matrix(np.hstack([b.reshape(-1, 1), -A]), number_type="float")
    lp_mat.rep_type = cdd.RepType.INEQUALITY
    lp_mat.obj_type = cdd.LPObjType.MAX
    lp_mat.obj_func = (0,) + tuple(d)
    lp = cdd.LinProg(lp_mat)
    lp.solve()

    return lp.obj_value


def compute_mrpi_hrep(Ak, W_A, W_b, epsilon=1e-4, max_iter=500):
    """
    Compute mRPI outer-approximation directly in H-rep (like MPT3).
    Returns (A_F, b_F).
    """
    nx = Ak.shape[0]
    s, alpha, Ms = 0, 1000, 1000

    # Step 1: find s such that alpha small enough
    while alpha > epsilon / (epsilon + Ms) and s < max_iter:
        s += 1
        dirs = (np.linalg.matrix_power(Ak.T, s) @ W_A.T).T
        alpha = np.max(
            [support_function_lp(W_A, W_b, d) / bi for d, bi in zip(dirs, W_b)]
        )

        mss = np.zeros((2 * nx, 1))
        for i in range(1, s + 1):
            mss += np.array(
                [
                    support_function(W_A, W_b, np.linalg.matrix_power(Ak, i)).reshape(
                        -1, 1
                    ),
                    support_function(W_A, W_b, -np.linalg.matrix_power(Ak, i)).reshape(
                        -1, 1
                    ),
                ]
            ).reshape((2 * nx, 1))

        Ms = max(mss)

        # # crude Ms bound
        # mss = []
        # for i in range(1, s+1):
        #     for d in [np.eye(nx)[j] for j in range(nx)]:
        #         mss.append(support_function_lp(W_A, W_b, np.linalg.matrix_power(Ak,i) @ d))
        #         mss.append(support_function_lp(W_A, W_b, -np.linalg.matrix_power(Ak,i) @ d))
        # Ms = max(mss) if mss else Ms

    # Step 2: build H-rep of F
    A_F = W_A.copy()
    b_F = []
    Fs = pytope.Polytope(A=W_A, b=W_b)
    for i in range(1, s):
        Fs = Fs + np.linalg.matrix_power(Ak, i) * Fs
    Fs = (1 / 1 - alpha) * Fs
    # for a, b in zip(W_A, W_b):
    #     # sum support values in direction (A^k)^T a
    #     s_val = 0.0
    #     for k in range(s):
    #         d = (np.linalg.matrix_power(Ak,k).T @ a)
    #         s_val += support_function_lp(W_A, W_b, d)
    #     b_F.append(s_val / (1 - alpha))
    # b_F = np.array(b_F)

    A_F = Fs.A
    b_F = Fs.b

    return A_F, b_F
