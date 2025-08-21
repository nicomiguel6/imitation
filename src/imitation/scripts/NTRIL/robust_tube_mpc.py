"""Robust Tube Model Predictive Control for data augmentation in NTRIL pipeline."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from scipy.optimize import minimize

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
        disturbance_bound: float = 0.1,
        tube_radius: float = 0.05,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linearization_method: str = "finite_difference",
        finite_diff_eps: float = 1e-6,
    ):
        """Initialize Robust Tube MPC.
        
        Args:
            horizon: MPC prediction horizon
            disturbance_bound: Bound for disturbance set
            tube_radius: Radius of the robust tube
            Q: State cost matrix (if None, will be identity)
            R: Control cost matrix (if None, will be identity)
            control_bounds: Tuple of (lower_bound, upper_bound) for controls
            linearization_method: Method for linearizing dynamics
            finite_diff_eps: Epsilon for finite difference linearization
        """
        self.horizon = horizon
        self.disturbance_bound = disturbance_bound
        self.tube_radius = tube_radius
        self.Q = Q
        self.R = R
        self.control_bounds = control_bounds
        self.linearization_method = linearization_method
        self.finite_diff_eps = finite_diff_eps
        
        # Storage for learned dynamics
        self.A_matrices: List[np.ndarray] = []
        self.B_matrices: List[np.ndarray] = []
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
    
    def fit_dynamics(
        self, 
        trajectories: List[types.Trajectory],
        method: str = "least_squares",
    ) -> Dict[str, Any]:
        """Fit linear dynamics model from trajectory data.
        
        Args:
            trajectories: List of trajectories to fit dynamics from
            method: Method for fitting dynamics ("least_squares", "ridge")
            
        Returns:
            Dictionary with fitting statistics
        """
        # Collect state-action-next_state tuples
        states = []
        actions = []
        next_states = []
        
        for traj in trajectories:
            for t in range(len(traj.obs) - 1):
                states.append(traj.obs[t])
                actions.append(traj.acts[t])
                next_states.append(traj.obs[t + 1])
        
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        
        self.state_dim = states.shape[1]
        self.action_dim = actions.shape[1]
        
        # Fit linear dynamics: x_{t+1} = A * x_t + B * u_t + w_t
        # Stack [states, actions] as input matrix
        inputs = np.hstack([states, actions])
        
        if method == "least_squares":
            # Solve least squares: inputs @ theta = next_states
            theta = np.linalg.lstsq(inputs, next_states, rcond=None)[0]
        elif method == "ridge":
            # Ridge regression for numerical stability
            lambda_reg = 1e-4
            theta = np.linalg.solve(
                inputs.T @ inputs + lambda_reg * np.eye(inputs.shape[1]),
                inputs.T @ next_states
            )
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        # Extract A and B matrices
        self.A_matrices = [theta[:self.state_dim].T]
        self.B_matrices = [theta[self.state_dim:].T]
        
        # Compute fitting error
        predicted_next_states = inputs @ theta
        mse = np.mean((next_states - predicted_next_states) ** 2)
        
        # Initialize cost matrices if not provided
        if self.Q is None:
            self.Q = np.eye(self.state_dim)
        if self.R is None:
            self.R = np.eye(self.action_dim)
        
        return {
            "mse": mse,
            "n_transitions": len(states),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
    
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
