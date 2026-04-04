"""Robust Tube Model Predictive Control for data augmentation in NTRIL pipeline."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from itertools import product
import os

import gymnasium as gym
import numpy as np
import torch as th
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are, solve_continuous_are
import casadi as ca
import do_mpc
import cdd
import pytope
import tqdm

from matplotlib import pyplot as plt

from imitation.data import types
from imitation.policies.base import NonTrainablePolicy
from imitation.util import util
from imitation.data import types
from imitation.scripts.NTRIL.double_integrator.double_integrator import generate_reference_trajectory


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
        use_approx: bool = False,
    ):
        """Initialize Robust Tube MPC.

        Args:
            horizon: MPC prediction horizon
            episode_length: Length of episode
            time_step: discretization time step for MPC
            disturbance_bound: Bound for disturbance set
            tube_radius: Radius of the robust tube
            A: State Dynamics matrix [MUST BE DISCRETE TIME]
            B: Control Dynamics matrix [MUST BE DISCRETE TIME]
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
        self.use_approx = use_approx
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

        # Tracks the number of solve_mpc() calls since the last reset_episode().
        # Used to force-sync mpc.t0 and simulator.t0 before every solve, making
        # time consistent with env.step_count regardless of what ran before.
        self._episode_step: int = 0

        # Nominal state z tracked separately from the true state x.
        # The MPC always solves from z (not x), so the predicted nominal
        # trajectory is undisturbed.  The ancillary control K@(x - z) then
        # drives the true state back toward z, keeping the error e = x - z in Z.
        # Initialised to None; set to the initial physical state in reset_episode().
        self._z_nominal: Optional[np.ndarray] = None

        if reference_trajectory is None: # default to all zeros if none is provided, this changes the cost function to a minimization problem instead of tracking problem
            self.reference_trajectory = types.Trajectory(obs=np.zeros((self.episode_length + 1, self.state_dim)), acts=np.zeros((self.episode_length, self.action_dim)), infos=np.array([{}] * self.episode_length), terminal=True)
        else:
            self.reference_trajectory = reference_trajectory
    
    def setup(self):
        """Initialize necessary objects for setting up MPC"""

        # ------------ UNDISTURBED MODEL SETUP ------------- #
        # for use in generating nominal control
        model_type = "discrete"
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
        # disc_model = nominal_model.discretize(t_step=self.time_step)
        # self.A_d = disc_model._A
        # self.B_d = disc_model._B
        self.A_d = self.A
        self.B_d = self.B

        # Solve for P, K matrices
        self.P = solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)
        self.K = -np.linalg.inv(self.B_d.T @ self.P @ self.B_d + self.R) @ (
            self.B_d.T @ self.P @ self.A_d
        )
        # Closed loop
        self.Ak = self.A_d + self.B_d @ self.K

        if self.disturbance_bound is not None:

            # Set up disturbance polytope
            self.disturbance_polytope = pytope.Polytope(self.disturbance_vertices)

            # Compute mrpi set
            if self.use_approx:
                self.Z = compute_approximate_linear_mrpi(self, disturbance_magnitude=self.disturbance_bound)
                self.A_F = self.Z.A
                self.b_F = self.Z.b
            else:
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
        # setup_mpc = {
        #     "n_horizon": self.horizon,
        #     "t_step": self.time_step,
        #     "state_discretization": "collocation",
        #     "collocation_type": "radau",
        #     "collocation_deg": 3,
        #     "collocation_ni": 2,
        #     "store_full_solution": True,
        # }
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': self.time_step,
            # 'state_discretization': 'discrete',
            'store_full_solution': True,
        }
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

        # Print the bounds
        if self.disturbance_bound is not None:
            print("---------------------------------------------------------")
            print("Lower bounds of the states: ", mpc.bounds["lower", "_x", "x"])
            print("Upper bounds of the states: ", mpc.bounds["upper", "_x", "x"])
            print("Lower bounds of the input: ", mpc.bounds["lower", "_u", "u"])
            print("Upper bounds of the input: ", mpc.bounds["upper", "_u", "u"])
        
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
            self.estimator = do_mpc.estimator.StateFeedback(disturbed_model)
        
        else:
            self.simulator = do_mpc.simulator.Simulator(nominal_model)
            self.estimator = do_mpc.estimator.StateFeedback(nominal_model)

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

    def _extract_physical_state(self, obs: np.ndarray) -> np.ndarray:
        """Return the physical state from a (possibly augmented) observation.

        The environment observation may be augmented with reference states,
        e.g. ``[pos, vel, ref_pos, ref_vel]``.  The MPC operates entirely in
        physical state space (first ``state_dim`` elements), so every time an observation flows into MPC machinery this method should be called first.
        """
        return np.asarray(obs).flatten()[:self.state_dim]

    def set_reference_trajectory(self, reference_trajectory: types.Trajectory) -> None:
        """Update the reference trajectory without rebuilding the MPC problem.

        The compiled NLP, MRPI sets, and all CasADi machinery are reused.
        Only the data read by tvp_fun at each solve step changes.

        Args:
            reference_trajectory: New reference trajectory to track.
        """
        self.reference_trajectory = reference_trajectory
        self._episode_step = 0
        self._z_nominal = None  # cleared; reset_episode() must be called next
        if self.mpc is not None:
            self.mpc.reset_history()
            self.mpc.t0 = 0.0
        if self.simulator is not None:
            self.simulator.reset_history()
            self.simulator.t0 = 0.0
        self.xs = []
        self.us = []

    def reset_episode(
        self,
        initial_state: np.ndarray,
        t_start: float = 0.0,
    ) -> None:
        """Reset MPC and simulator for a new episode (or after a time jump).

        Must be called:
        - At the start of every gymnasium episode, before the first solve_mpc().
        - After set_reference_trajectory(), once the initial state is known.
        - When resuming mid-trajectory: pass t_start = env.step_count * time_step.

        This is the *only* place set_initial_guess() is called.  All subsequent
        solve_mpc() calls warm-start from the previous step's NLP solution.

        Args:
            initial_state: Initial physical state (or augmented obs — the physical
                           slice is extracted automatically).
            t_start: Starting time in seconds.  Default 0.0 for a fresh episode.
                     Pass env.step_count * time_step to resume mid-trajectory.
        """
        phys = np.asarray(initial_state).flatten()[:self.state_dim].reshape(-1, 1)
        self._episode_step = int(round(t_start / self.time_step))

        if self.mpc is not None:
            self.mpc.reset_history()
            self.mpc.t0 = t_start
            self.mpc.x0 = phys
            self.mpc.set_initial_guess()

        if self.simulator is not None:
            self.simulator.reset_history()
            self.simulator.t0 = t_start
            self.simulator.x0 = phys

        if hasattr(self, "estimator") and self.estimator is not None:
            self.estimator.x0 = phys

        # At episode start the nominal trajectory begins at the true state,
        # so the initial error e_0 = x_0 - z_0 = 0.
        self._z_nominal = phys.flatten().copy()

        self.xs = []
        self.us = []

    def solve_to_end(self, initial_state:np.ndarray, n_steps:int) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the MPC problem to the end of the horizon.

        Args:
            initial_state: Initial state of the system.
            n_steps: Number of steps to solve the MPC problem.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (next_state, applied_control).
        """

        self.mpc.reset_history()
        self.mpc.t0 = 0.0
        self.simulator.t0 = 0.0
        self._episode_step = 0

        self.mpc.x0 = initial_state
        self.estimator.x0 = initial_state
        self.simulator.x0 = initial_state
        self.mpc.set_initial_guess()

        x0 = initial_state
        xs = [x0]
        us = []
        for i in range(n_steps):
            u0 = self.mpc.make_step(x0)
            nominal_state = self.mpc.data.prediction(("_x", "x"))
            y0 = self.simulator.make_step(u0)
            x0 = self.estimator.make_step(y0)
            xs.append(x0.flatten())
            us.append(u0.flatten())
        
        x_actual = self.mpc.data["_x"]
        u_actual = self.mpc.data["_u"]
        
        return xs, us, x_actual, u_actual
    
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
                "Must call setup() and reset_episode() before solve_mpc()."
            )
        if self._z_nominal is None:
            raise ValueError(
                "reset_episode() must be called before the first solve_mpc()."
            )

        # ---- extract physical state (strip reference augmentation if present) ----
        x_true = np.asarray(self._extract_physical_state(state) if state.shape[0] > self.state_dim
                            else state).reshape(-1, 1)

        # ---- force-sync times (Fix 2) ----------------------------------------
        # Keeps t_idx in tvp_fun aligned with env.step_count and corrects any
        # drift from augment_trajectory() calls between solves.
        t_now = self._episode_step * self.time_step
        self.mpc.t0 = t_now
        self.simulator.t0 = t_now

        # ---- nominal state (Fix 4) -------------------------------------------
        # The MPC solves from z_nominal, NOT from x_true.  Because z propagates
        # under undisturbed dynamics, the predicted trajectory stays inside the
        # tightened constraints.  If we used x_true here the "nominal" would equal
        # the true state and the ancillary correction would always be zero.
        z = self._z_nominal.reshape(-1, 1)

        self.mpc.x0 = z
        self.simulator.x0 = x_true
        # set_initial_guess() is NOT called here (Fix 3): warm-start from the
        # previous step's NLP solution (initialised once in reset_episode()).

        # ---- solve MPC from nominal state ------------------------------------
        u_nom = self.mpc.make_step(z)

        # ---- ancillary correction u = u_nom + K(x_true - z) -----------------
        # Non-zero whenever disturbances have pushed x_true away from z.
        # K is the LQR gain for the closed-loop A_k = A_d + B_d K, which is
        # stable and keeps e = x_true - z inside the MRPI set Z.
        if self.disturbance_bound is not None:
            applied_u = u_nom + self.K @ (x_true - z)
        else:
            applied_u = u_nom

        # ---- propagate nominal state under undisturbed dynamics --------------
        # z_{t+1} = A_d z_t + B_d u_nom  (no disturbance term)
        self._z_nominal = (self.A_d @ z + self.B_d @ u_nom).flatten()

        # ---- simulate next true state (do-mpc internal simulator) -----------
        next_state = self.simulator.make_step(u0=applied_u)

        # ---- bookkeeping ----
        nominal_horizon = self.mpc.data.prediction(("_x", "x"))
        self._episode_step += 1
        self.xs.append(x_true)
        self.us.append(applied_u)

        return next_state, applied_u.flatten(), nominal_horizon

    def augment_trajectory(
        self,
        trajectory: types.Trajectory,
        partial_horizon: int = 50,
        k_timesteps: int = 5,
        n_augmentations: int = 5,
        reference_states: Optional[np.ndarray] = None,
        simulate_to_end: bool = False,
        initial_state_only: bool = False,
    ) -> Sequence[types.TrajectoryWithRew]:
        """Augment a trajectory using robust tube MPC. Sample points every k_timesteps and propagate partial trajectories following ancillary controller u = u0 + K(x-x0) .

        Args:
            trajectory: Trajectory to augment
            partial_horizon: Length of partial trajectory (ignored when simulate_to_end=True
                or initial_state_only=True)
            k_timesteps: Interval of time steps to select sample from (ignored when
                initial_state_only=True)
            n_augmentations: Number of augmented samples per transition
            reference_states: Reference states to append to the augmented trajectories
            simulate_to_end: If True, each sampled trajectory is propagated until the
                end of the episode rather than for a fixed partial_horizon.  Every
                timestep that is a multiple of k_timesteps is eligible regardless of
                how many steps remain.
            initial_state_only: If True, only t=0 is processed and each sample is
                propagated until the end of the episode (T steps).  Overrides both
                simulate_to_end and the k_timesteps eligibility check.
        Returns:
            Collection of partial trajectories sampled from RTMPC trajectory
        """

        augmented_infos = []
        augmented_trajs = []

        # Snapshot the simulator's time before augmentation so we can restore it
        # afterward.  augment_trajectory() calls simulator.make_step() many times,
        # which advances simulator.t0.  Restoring here means the method is
        # non-destructive to the episode state: a subsequent solve_mpc() call will
        # still see the correct t_now without relying on _episode_step to override it.
        _saved_sim_t0 = float(self.simulator.t0)

        n_obs = len(trajectory.obs)

        for t in range(n_obs - 1):
            # Determine number of simulation steps and eligibility for this timestep.
            if initial_state_only:
                eligible = t == 0
                n_steps = n_obs - 1  # always simulate the full episode from t=0
            elif simulate_to_end:
                n_steps = n_obs - 1 - t  # steps from t to the last obs
                eligible = t % k_timesteps == 0 and n_steps > 0
            else:
                n_steps = partial_horizon - 1
                eligible = t % k_timesteps == 0 and t + partial_horizon < n_obs

            if not eligible:
                continue

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

            # Simulate a partial trajectory for each sample.
            # The simulator's t0 is reset to t * time_step at the start of each
            # sample so every sample in the same outer iteration starts from the
            # same reference-trajectory index.
            for sample in samples:

                # Reset simulator time to the nominal time of this outer step
                # so the TVP (reference look-up) is consistent across samples.
                self.simulator.t0 = t * self.time_step

                # initialize trajectory builder
                builder = util.TrajectoryBuilder()
                sampled_reference_state = reference_states[t]
                builder.start_episode(initial_obs=np.concatenate((sample.flatten(), sampled_reference_state.flatten()), axis=0))

                # initialize state and action
                x = sample
                u = current_nominal_action
                self.simulator.x0 = x

                # Track nominal state/action locally so they can be updated per step
                step_nominal_state = current_nominal_state
                step_nominal_action = current_nominal_action

                # propagate dynamics
                for t_step in range(n_steps):
                    u_applied = step_nominal_action + self.K @ (
                        x.T.reshape(
                            -1, 1
                        )
                        - step_nominal_state.reshape(
                            -1, 1
                        )
                    )
                    x_next = self.simulator.make_step(u0=u_applied)
                    
                    # calculate tracking cost
                    tracking_cost = self._compute_tracking_cost_metric(
                        self._extract_physical_state(trajectory.obs[t + t_step]), x, trajectory.acts[t + t_step], u_applied, t + t_step
                    )
                    # calculate margin to safety violation
                    margin_safety = self._compute_margin_safety_violation(x)
                    x = x_next

                    if reference_states is not None:
                        x_next = np.concatenate((x_next, reference_states[t + t_step].reshape(-1, 1)), axis=0)
                    
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

                    step_nominal_state = trajectory.obs[t + t_step]
                    step_nominal_action = trajectory.acts[t + t_step]

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

            # # FOR DEBUG PURPOSES: PLOT SAMPLES, ADJUSTED SET, AND CURRENT NOMINAL STATE
            # fig, ax = plt.subplots()
            # ts = np.array(samples)
            # ax.plot(ts[:, 0], ts[:, 1], 'r+', label="Samples")
            # ax.plot(approx_tube.V[:, 0], approx_tube.V[:, 1], 'k+', label="Adjusted Set")
            # ax.plot(current_nominal_state[0], current_nominal_state[1], 'ro', label="Current Nominal State")
            # ax.legend()
            # os.makedirs("imitation/scripts/NTRIL/debug/double_integrator/debug/plots/samples", exist_ok=True)
            # plt.savefig(os.path.join("imitation/scripts/NTRIL/debug/double_integrator/debug/plots/samples", f"samples_and_adjusted_set_{t}.png"))
            # plt.close()

            # # FOR DEBUG PURPOSES: PLOT CURRENT REFERENCE NOISY ROLLOUT AND ALL AUGMENTATIONS FOR EACH TIMESTEP
            # fig, ax = plt.subplots()
            # ax.plot(trajectory.obs[:, 0], label="Reference Noisy Rollout")
            # timespan = np.arange(t*self.time_step, (t+n_steps+1)*self.time_step, self.time_step)
            # for traj in augmented_trajs_t:
            #     ax.plot(timespan, traj.obs[:, 0], label="Augmentation")
            # ax.legend()
            # os.makedirs("imitation/scripts/NTRIL/debug/double_integrator/debug/plots/augmented_trajectories", exist_ok=True)
            # plt.savefig(os.path.join("imitation/scripts/NTRIL/debug/double_integrator/debug/plots/augmented_trajectories", f"augmented_trajectory_{t}.png"))
            # plt.close()
        
        
        # Restore simulator time so this call is non-destructive to episode state.
        self.simulator.t0 = _saved_sim_t0

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


def compute_approximate_linear_mrpi(robust_mpc: RobustTubeMPC, disturbance_magnitude: float = 0.1):
    """Compute approximate mRPI set using monte carlo"""

    # Set up double integrator environment
    env = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0", disturbance_magnitude=disturbance_magnitude, dt = robust_mpc.time_step)

    n_trajectories = 100
    total_time_steps = 500
    trajectories = [[] for _ in range(n_trajectories)]
    min_vals = [[np.inf for _ in range(env.observation_space.shape[0])] for _ in range(n_trajectories)]
    max_vals = [[-np.inf for _ in range(env.observation_space.shape[0])] for _ in range(n_trajectories)]

    internal_state_dim = int(env.observation_space.shape[0]/2)



    # Set up ancillary controller
    K = robust_mpc.K

    # closed loop dynamics
    Ak = robust_mpc.A_d + robust_mpc.B_d @ K

    # Propagate trajectories forward
    for traj_idx in range(n_trajectories):
        # Set up zero initial state
        initial_state = np.array([0.0, 0.0]).reshape(-1, 1)
        trajectories[traj_idx].append(initial_state)
        
        # set initial state to zero
        obs, info = env.reset(state=initial_state.flatten())
        obs = obs[:internal_state_dim]
        for _ in range(total_time_steps):
            disturbance = np.random.uniform(-disturbance_magnitude, disturbance_magnitude, size=(internal_state_dim,))
            # get next state
            next_obs = robust_mpc.Ak @ obs.reshape(-1, 1) + disturbance.reshape(-1, 1)
            trajectories[traj_idx].append(next_obs.flatten())
            obs = next_obs.flatten()

            # update min and max values for each state
            for i in range(internal_state_dim):
                if next_obs[i] < min_vals[traj_idx][i]:
                    min_vals[traj_idx][i] = next_obs[i]
                if next_obs[i] > max_vals[traj_idx][i]:
                    max_vals[traj_idx][i] = next_obs[i]
    
    # Compute initial mRPI set (minimum and maximum values for each state across all trajectories) and inflate by 5-10%
    initial_mrpis = []
    for i in range(internal_state_dim):
        min_val = np.min([min_vals[traj_idx][i] for traj_idx in range(n_trajectories)])
        max_val = np.max([max_vals[traj_idx][i] for traj_idx in range(n_trajectories)])
        initial_mrpis.append([min_val, max_val])
    
    # Verify invariance: for every corner of Z-hat x W, check A_K x + w stays inside Z-hat.
    # Rebuild samples and retry whenever any corner violates the box — inflation can make
    # previously-passing corners violate again, so we must complete a clean pass each time.
    state_vertices = list(product(*[[lo,hi] for lo, hi in initial_mrpis]))
    dist_vertices = list(product(*[[-disturbance_magnitude, disturbance_magnitude] for _ in range(internal_state_dim)]))

    all_inside = False
    while not all_inside:
        all_inside = True
        # track worst violation
        worst_low = [initial_mrpis[i][0] for i in range(internal_state_dim)]
        worst_high = [initial_mrpis[i][1] for i in range(internal_state_dim)]

        for state_corner in state_vertices:
            for dist_corner in dist_vertices:
                prop = robust_mpc.Ak @ np.array(state_corner).reshape(-1, 1) + np.array(dist_corner).reshape(-1, 1)
                for i in range(internal_state_dim):
                    if prop[i] < initial_mrpis[i][0] or prop[i] > initial_mrpis[i][1]:
                        all_inside = False
                    
                    worst_low[i] = min(worst_low[i], prop[i])
                    worst_high[i] = max(worst_high[i], prop[i])
                        
        if not all_inside:
            for i in range(internal_state_dim):
                box_size = initial_mrpis[i][1] - initial_mrpis[i][0]
                margin = 0.01 * max(box_size, 1e-6)
                initial_mrpis[i][0] = min(initial_mrpis[i][0], worst_low[i] - margin)
                initial_mrpis[i][1] = max(initial_mrpis[i][1], worst_high[i] + margin)
            
            # state_vertices = list(product(*[[lo,hi] for lo, hi in initial_mrpis]))

    # convert to points for pytope.Polytope
    initial_mrpis = np.array(initial_mrpis).reshape(-1, internal_state_dim)
    nd_array = np.array([initial_mrpis[i] for i in range(internal_state_dim)])
    grid_points = np.meshgrid(*nd_array)
    initial_mrpi_points = np.vstack(grid_points[i].flatten() for i in range(internal_state_dim)).T
    return pytope.Polytope(initial_mrpi_points)

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

                sample_min = np.array(midpoints.copy())
                sample_min[j_dim] = minpoints[j_dim]
                samples.append(sample_min)

                sample_max = np.array(midpoints.copy())
                sample_max[j_dim] = maxpoints[j_dim]
                samples.append(sample_max)

    return samples


def get_vertices(A, b):
    """Converts H-rep into a list of vertices"""

    # Form h-matrix and extract generators
    mat = cdd.Matrix(np.hstack([b.reshape(-1, 1), -A]), number_type="fraction")
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

    lp_mat = cdd.Matrix(np.hstack([b.reshape(-1, 1), -A]), number_type="fraction")
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
    Fs = (1 / (1 - alpha)) * Fs
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


if __name__ == "__main__": # test code
    import matplotlib.pyplot as plt

    disturbance_magnitude = 0
    disturbance_vertices = np.array([[disturbance_magnitude, disturbance_magnitude], [-disturbance_magnitude, -disturbance_magnitude], [-disturbance_magnitude, disturbance_magnitude], [disturbance_magnitude, -disturbance_magnitude]])
    
    dt = 0.1
    # Set up double integrator environment
    env = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0", max_episode_seconds=200.0, dt = dt, disturbance_magnitude=disturbance_magnitude)
    # Set up reference trajectory
    reference_trajectory = generate_reference_trajectory(T=env.max_episode_steps, dt=dt, mode="sinusoidal", amplitude=1.0, frequency=0.1, phase=0.0)
    reference_trajectory_mpc = types.Trajectory(obs=reference_trajectory, acts=np.zeros((env.max_episode_steps, 1)), infos=np.array([{}] * env.max_episode_steps), terminal=True)

    # Set up robust tube MPC for env usage
    robust_mpc = RobustTubeMPC(
        horizon = 10,
        time_step = dt,
        A = env.A_d,
        B = env.B_d,
        Q = np.diag([10.0, 1.0]),
        R = 0.1*np.eye(1),
        # disturbance_bound = disturbance_magnitude,
        # disturbance_vertices = disturbance_vertices,
        state_bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds = (np.array([-20.0]), np.array([20.0])),
        reference_trajectory = reference_trajectory_mpc,
        use_approx = True,
    )

    # set up identical robust tube MPC for internal usage
    robust_mpc_internal = RobustTubeMPC(
        horizon = 10,
        time_step = dt,
        A = env.A_d,
        B = env.B_d,
        Q = np.diag([10.0, 1.0]),
        R = 0.1*np.eye(1),
        # disturbance_bound = disturbance_magnitude,
        # disturbance_vertices = disturbance_vertices,
        state_bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds = (np.array([-20.0]), np.array([20.0])),
        reference_trajectory = reference_trajectory_mpc,
        use_approx = True,
    )

    robust_mpc.setup()
    robust_mpc_internal.setup()

    # Compute approximate linear mRPI set
    # approximate_linear_mrpi = compute_approximate_linear_mrpi(robust_mpc, disturbance_magnitude=disturbance_magnitude)

    # Run MPC for 100 steps and plot
    initial_state = np.array([-5.0, -2.0], dtype=np.float32)
    obs, info = env.reset(
        state=initial_state,
        options={"reference_trajectory": reference_trajectory},
    )

    robust_mpc.set_reference_trajectory(reference_trajectory_mpc)
    robust_mpc_internal.set_reference_trajectory(reference_trajectory_mpc)

    # check initial observation is in tightened state bounds
    # tightened_set = pytope.Polytope(A=robust_mpc.A_x_t, b=robust_mpc.b_x_t)
    # while not tightened_set.contains(obs[:robust_mpc.state_dim]):
    #     obs, info = env.reset(options={"reference_trajectory": reference_trajectory})

    states = []
    actions = []
    nominal_states_trajectories = []

    # --- Option A: gymnasium loop (env drives the dynamics) ---
    # reset_episode() must be called once before the first solve_mpc() to
    # initialise the solver, set t0 = 0 on both MPC and simulator, and record
    # the initial_guess.  Subsequent solve_mpc() calls warm-start from here.
    robust_mpc.reset_episode(obs[:robust_mpc.state_dim])
    for _ in range(env.max_episode_steps):
        next_state, action, nominal_state = robust_mpc.solve_mpc(obs)
        # compare env time with mpc time
        # print(f"env time: {env.step_count * env.dt}, mpc time: {robust_mpc.mpc.t0}")
        action = action.flatten()
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        actions.append(action)
        nominal_states_trajectories.append(nominal_state)

    # --- Option B: internal loop (do-mpc simulator drives the dynamics) ---
    xs, us, x_actual, u_actual = robust_mpc_internal.solve_to_end(initial_state, env.max_episode_steps)
    states_internal = xs
    actions_internal = us

    # # Test env step()
    # for itr in range(env.max_episode_steps):
    #     action = actions_internal[itr]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     states.append(obs)
    #     actions.append(action)

    # Plot states and controls in separate subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot state trajectories (position and velocity)
    axs[0].plot(np.array(states)[:, 0], 'r', label="Position (env)")
    # axs[0].plot(np.array(states)[:, 1], 'b', label="Velocity (env)")
    axs[0].plot(np.array(states_internal)[:, 0], color="#FF7F7F", label="Position (internal)")
    #axs[0].plot(np.array(states_internal)[:, 1], color="#7FBFFF", label="Velocity (internal)")
    axs[0].plot(reference_trajectory[:, 0], 'r-.', label="Reference Position")
    #axs[0].plot(reference_trajectory[:, 1], 'b-.', label="Reference Velocity")
    axs[0].axhline(y=-10.0, color="k", linestyle="--", linewidth=1)
    axs[0].axhline(y=10.0, color="k", linestyle="--", linewidth=1)
    axs[0].set_ylabel("State")
    axs[0].set_title("State Trajectories")
    axs[0].legend(loc="best")

    # Plot control action
    axs[1].plot(np.arange(len(actions)), np.array(actions)[:, 0], 'g', label="Control (env)")
    axs[1].plot(np.arange(len(actions_internal)), np.array(actions_internal)[:, 0], 'k', label="Control (internal)")
    axs[1].axhline(y=-5.0, color="k", linestyle="--", linewidth=1)
    axs[1].axhline(y=5.0, color="k", linestyle="--", linewidth=1)
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel("Control")
    axs[1].set_title("Control Action")
    axs[1].legend(loc="best")

    plt.tight_layout()
    plt.legend()
    plt.show()

    a = 5
