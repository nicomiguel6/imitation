"""Robust Tube Model Predictive Control for data augmentation in NTRIL pipeline."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are
import casadi as ca
import do_mpc
import cdd
import pytope

from imitation.data import types
from imitation.util import util


class RobustTubeMPC:
    """Robust Tube Model Predictive Control for trajectory augmentation.
    
    This class implements linear robust tube MPC under a specified disturbance set
    to augment state-action pairs using Tagliabue's method.
    """
    
    def __init__(
        self,
        horizon: int = 10,
        time_step: float = 0.1,
        disturbance_bound: float = 0.1,
        tube_radius: float = 0.05,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        disturbance_vertices: Optional[np.ndarray] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linearization_method: str = "finite_difference",
        finite_diff_eps: float = 1e-6,
    ):
        """Initialize Robust Tube MPC.
        
        Args:
            horizon: MPC prediction horizon
            time_step: discretization time step
            disturbance_bound: Bound for disturbance set
            tube_radius: Radius of the robust tube
            A: State Dynamics matrix
            B: Control Dynamics matrix
            Q: State cost matrix (if None, will be identity)
            R: Control cost matrix (if None, will be identity)
            control_bounds: Tuple of (lower_bound, upper_bound) for controls
            linearization_method: Method for linearizing dynamics
            finite_diff_eps: Epsilon for finite difference linearization
        """
        self.horizon = horizon
        self.time_step = time_step
        self.disturbance_bound = disturbance_bound
        self.tube_radius = tube_radius
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.disturbance_vertices = disturbance_vertices
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds
        self.linearization_method = linearization_method
        self.finite_diff_eps = finite_diff_eps
        
        self.state_dim = np.shape(A)[1]
        self.action_dim = np.shape(B)[1]
    
    def setup(self):

        # ------------ PRELIMINARY SETS ------------- #
        # Solve for P, K matrices
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = -np.linalg.inv(self.B.T @ self.P @ self.B + self.R) @ (self.B @ self.P @ self.A)

        # Closed loop
        self.Ak = self.A + self.B@self.K

        # Set up disturbance polytope
        self.disturbance_polytope = pytope.Polytope(self.disturbance_vertices)

        # Compute mrpi set
        self.A_F, self.b_F = compute_mrpi_hrep(self.Ak, self.disturbance_polytope.A, self.disturbance_polytope.b)

        # Tighten state and constraint bounds accordingly
        if self.state_bounds is not None:
            A_x_t, b_x_t = tighten_state_constraints(self.state_bounds, self.A_F, self.b_F)
        
        if self.control_bounds is not None:
            A_u_t, b_u_t = tighten_control_constraints(self.control_bounds, self.K, self.A_F, self.b_F)

        # ------------ UNDISTURBED MODEL SETUP ------------- #
        # for use in generating nominal control
        model_type = 'discrete'
        nominal_model = do_mpc.model.Model(model_type)

        _x = nominal_model.set_variable(var_type='_x', var_name='x', shape=(self.state_dim,1))
        _u = nominal_model.set_variable(var_type='_u', var_name='u', shape=(self.action_dim,1))

        nominal_x_next = self.A@_x + self.B@_u 

        nominal_model.set_rhs('x', nominal_x_next)

        nominal_model.setup()

        # ------------ CONTROLLER SETUP ------------- #
        # solver for nominal control
        mpc = do_mpc.controller.MPC(nominal_model)
        setup_mpc = {'n_horizon': self.horizon,
                     't_step': self.time_step,
                     'state_dicretization': 'discrete',
                     'store_full_solution': True}
        
        # currently assumes all constraints are simple bounding boxes
        if self.state_bounds is not None: 
            # lower bounds of the states
            mpc.bounds['lower','_x','x'] = np.array([-b_x_t[1], -b_x_t[3]])

            # upper bounds of the states
            mpc.bounds['upper','_x','x'] = np.array([b_x_t[0], b_x_t[2]])

        if self.control_bounds is not None:
            # lower bounds of the input
            mpc.bounds['lower','_u','u'] = -np.array([b_u_t[0]])

            # upper bounds of the input
            mpc.bounds['upper','_u','u'] =  np.array([b_u_t[1]])
        
        mpc.set_param(**setup_mpc)

        # Objective
        x = nominal_model.x['x']
        u = nominal_model.u['u']

        lterm = (x.T @ self.Q @ x)
        mterm = x.T @ self.P @ x

        mpc.settings.set_linear_solver()
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(ca.SX(self.R))

        mpc.setup()

        # ------------ DISTURBED MODEL & SIMULATOR SETUP ------------- #        
        disturbed_model = do_mpc.model.Model(model_type)

        _xd = disturbed_model.set_variable(var_type='_x', var_name='xd', shape=(self.state_dim,1))
        _ud = disturbed_model.set_variable(var_type='_u', var_name='ud', shape=(self.action_dim,1))

        # Set disturbance
        _d = disturbed_model.set_variable(var_type='_p', var_name='d', shape=(self.state_dim,1))

        disturbed_x_next = self.A@_xd + self.B@_ud + _d
        disturbed_model.setup()

        self.simulator = do_mpc.simulator.Simulator(disturbed_model)

        d_template = self.simulator.get_p_template()

        def d_fun(t_now):
            d_template['d'] = sample_from_disturbance(self.disturbance_polytope)
            return d_template
        
        self.simulator.set_p_fun(d_fun)

        self.simulator.set_param(t_step = self.time_step)
        self.simulator.setup()

        print("Nominal and disturbed models, controller, and simulator setup completed!")

    
    def augment_trajectory(
        self, 
        trajectory: types.Trajectory,
        noise_level: float,
        n_augmentations: int = 5,
    ) -> types.Transitions:
        """Augment a trajectory using robust tube MPC.
        
        Args:
            trajectory: Trajectory to augment
            noise_level: Noise level used to generate this trajectory
            n_augmentations: Number of augmented samples per transition
            
        Returns:
            Augmented transitions
        """
        if not self.A_matrices or not self.B_matrices:
            raise ValueError("Dynamics must be fitted before augmentation")
        
        augmented_obs = []
        augmented_acts = []
        augmented_next_obs = []
        augmented_dones = []
        augmented_infos = []
        
        A = self.A_matrices[0]  # Use first (and only) fitted matrix
        B = self.B_matrices[0]
        
        for t in range(len(trajectory.obs) - 1):
            current_state = trajectory.obs[t]
            current_action = trajectory.acts[t]
            next_state = trajectory.obs[t + 1]
            
            # Generate augmented states around current state
            for _ in range(n_augmentations):
                # Sample disturbance for current state
                disturbance = self._sample_disturbance()
                augmented_state = current_state + disturbance
                
                # Solve robust tube MPC from augmented state
                optimal_action = self._solve_tube_mpc(
                    augmented_state, 
                    trajectory.obs[t:],  # Remaining trajectory as reference
                    noise_level
                )
                
                # Predict next state using learned dynamics
                predicted_next_state = A @ augmented_state + B @ optimal_action
                
                # Add small noise to next state to account for model uncertainty
                next_state_noise = self._sample_disturbance(scale=0.5)
                augmented_next_state = predicted_next_state + next_state_noise
                
                augmented_obs.append(augmented_state)
                augmented_acts.append(optimal_action)
                augmented_next_obs.append(augmented_next_state)
                augmented_dones.append(False)
                augmented_infos.append({
                    "original_timestep": t,
                    "noise_level": noise_level,
                    "augmentation_method": "robust_tube_mpc",
                })
        
        return types.Transitions(
            obs=np.array(augmented_obs),
            acts=np.array(augmented_acts),
            next_obs=np.array(augmented_next_obs),
            dones=np.array(augmented_dones),
            infos=np.array(augmented_infos),
        )
    
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
    
    def _solve_tube_mpc(
        self,
        initial_state: np.ndarray,
        reference_trajectory: np.ndarray,
        noise_level: float,
    ) -> np.ndarray:
        """Solve robust tube MPC optimization problem.
        
        Args:
            initial_state: Initial state for MPC
            reference_trajectory: Reference trajectory to follow
            noise_level: Current noise level (affects tube size)
            
        Returns:
            Optimal first control action
        """
        if not self.A_matrices or not self.B_matrices:
            raise ValueError("Dynamics must be fitted")
        
        A = self.A_matrices[0]
        B = self.B_matrices[0]
        
        # Adjust tube radius based on noise level
        current_tube_radius = self.tube_radius * (1 + noise_level)
        
        # MPC horizon (limited by reference trajectory length)
        mpc_horizon = min(self.horizon, len(reference_trajectory))
        
        # Decision variables: [u_0, u_1, ..., u_{N-1}]
        n_vars = self.action_dim * mpc_horizon
        
        def objective(u_vec):
            """MPC objective function."""
            u_sequence = u_vec.reshape((mpc_horizon, self.action_dim))
            
            cost = 0.0
            state = initial_state.copy()
            
            for k in range(mpc_horizon):
                # State cost (tracking reference)
                if k < len(reference_trajectory):
                    ref_state = reference_trajectory[k]
                    state_error = state - ref_state
                    cost += state_error.T @ self.Q @ state_error
                
                # Control cost
                control = u_sequence[k]
                cost += control.T @ self.R @ control
                
                # Predict next state
                if k < mpc_horizon - 1:
                    state = A @ state + B @ control
            
            return cost
        
        def constraint_func(u_vec):
            """Constraint function for tube constraints."""
            u_sequence = u_vec.reshape((mpc_horizon, self.action_dim))
            
            constraints = []
            state = initial_state.copy()
            
            for k in range(mpc_horizon):
                # Tube constraint: ||x_k - x_ref_k|| <= tube_radius
                if k < len(reference_trajectory):
                    ref_state = reference_trajectory[k]
                    state_error_norm = np.linalg.norm(state - ref_state)
                    constraints.append(current_tube_radius - state_error_norm)
                
                # Predict next state
                if k < mpc_horizon - 1:
                    control = u_sequence[k]
                    state = A @ state + B @ control
            
            return np.array(constraints)
        
        # Initial guess (zero controls)
        u0 = np.zeros(n_vars)
        
        # Control bounds
        bounds = None
        if self.control_bounds is not None:
            lower_bounds = np.tile(self.control_bounds[0], mpc_horizon)
            upper_bounds = np.tile(self.control_bounds[1], mpc_horizon)
            bounds = list(zip(lower_bounds, upper_bounds))
        
        # Solve optimization
        constraints = {'type': 'ineq', 'fun': constraint_func}
        
        try:
            result = minimize(
                objective,
                u0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                optimal_controls = result.x.reshape((mpc_horizon, self.action_dim))
                return optimal_controls[0]  # Return first control action
            else:
                # Fallback: return zero control
                return np.zeros(self.action_dim)
                
        except Exception:
            # Fallback: return zero control
            return np.zeros(self.action_dim)
    
    def get_tube_statistics(
        self, 
        trajectories: List[types.Trajectory],
    ) -> Dict[str, Any]:
        """Compute statistics about tube constraints satisfaction.
        
        Args:
            trajectories: Trajectories to analyze
            
        Returns:
            Dictionary with tube statistics
        """
        if not trajectories:
            return {}
        
        deviations = []
        for traj in trajectories:
            for t in range(len(traj.obs) - 1):
                # Compute deviation from nominal trajectory
                # (This is a simplified version - in practice you'd have a reference)
                if t > 0:
                    deviation = np.linalg.norm(traj.obs[t] - traj.obs[t-1])
                    deviations.append(deviation)
        
        if deviations:
            return {
                "mean_deviation": np.mean(deviations),
                "max_deviation": np.max(deviations),
                "std_deviation": np.std(deviations),
                "tube_violations": sum(d > self.tube_radius for d in deviations),
                "violation_rate": sum(d > self.tube_radius for d in deviations) / len(deviations),
            }
        else:
            return {"mean_deviation": 0.0, "max_deviation": 0.0, "std_deviation": 0.0}



def tighten_state_constraints(state_constraint_vertices: List[np.ndarray], A_F: np.ndarray, b_F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def tighten_control_constraints(control_constraint_vertices: np.ndarray, K: np.ndarray, A_F: np.ndarray, b_F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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




def get_vertices(A, b):
    """ Converts H-rep into a list of vertices """

    # Form h-matrix and extract generators
    mat = cdd.Matrix(np.hstack([b.reshape(-1,1), -A]), number_type='float')
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
    d = np.atleast_2d(d)      # shape (m, n_dirs)
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
        h = support_function_lp(A_F, b_F, a)   # your routine
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
        d = K.T @ a.reshape(-1,1)               # map direction
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
        A.append(e_i); b.append(upper[i])
        A.append(-e_i); b.append(-lower[i])

    return np.vstack(A), np.array(b)

def support_function_lp(A, b, d):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    h_repr = np.hstack([b.reshape(-1,1)])

    lp_mat = cdd.Matrix(np.hstack([b.reshape(-1,1), -A]), number_type='float')
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
    while alpha > epsilon/(epsilon + Ms) and s < max_iter:
        s += 1
        dirs = (np.linalg.matrix_power(Ak, s) @ W_A.T).T
        alpha = np.max([
            support_function_lp(W_A, W_b, d) / bi
            for d, bi in zip(dirs, W_b)
        ])

        mss = np.zeros((2*nx, 1))
        for i in range(1, s+1):
            mss += np.array([support_function(W_A, W_b, np.linalg.matrix_power(Ak, i)).reshape(-1, 1), support_function(W_A, W_b, -np.linalg.matrix_power(Ak, i)).reshape(-1, 1)]).reshape((2*nx, 1))
        
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
    Fs = pytope.Polytope(A = W_A, b = W_b)
    for i in range (1,s):
        Fs = Fs + np.linalg.matrix_power(Ak, i) * Fs
    Fs = (1/1-alpha)*Fs
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