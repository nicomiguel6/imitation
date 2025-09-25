"""Code for testing and debugging NTRIL. Step 3: Robust Tube MPC."""

import numpy as np
import os
import gymnasium as gym
import osqp
import hypothesis
import hypothesis.strategies as st
import casadi as ca
import do_mpc
import cdd
from scipy.linalg import solve_discrete_are

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.scripts.NTRIL.ntril import NTRILTrainer
from imitation.scripts.NTRIL.utils import visualize_noise_levels, analyze_ranking_quality
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import logger, util
from imitation.util.logger import configure
from imitation.data import serialize


def main():

    # Set up discrete linear system
    A = np.array([[1,1],[0,1]])
    B = np.array([[0.5], [1]])
    Q = np.diag([1,1])
    R = 0.1

    P = solve_discrete_are(A, B, Q, R)

    # Set up disturbance with cdd
    W_vertex = np.array([[0.15, 0.15],[0.15, -0.15],[-0.15,-0.15],[-0.15,0.15]], dtype=object)
    gen_mat = cdd.Matrix(
        np.hstack([np.ones((W_vertex.shape[0], 1), dtype=object), W_vertex]).tolist(),
        number_type='fraction'
    )
    gen_mat.rep_type = cdd.RepType.GENERATOR

    W = cdd.Polyhedron(gen_mat)
    W_polyhedron = Polyhedron(V = W_vertex)

    # Set up discrete disturbed linear system
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    _x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    x_next = A@_x + B@_u # linear system without disturbance

    model.set_rhs('x', x_next)

    model.setup()

    # Controller
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {'n_robust': 0,
                 'n_horizon': 7,
                 't_step': 0.5,
                 'state_discretization': 'discrete',
                 'store_full_solution': True,
                 'nplsol_opts': {'ipopt.linear_solver': 'MA27'}}
    
    mpc.set_param(**setup_mpc)

    # Objective
    x = model.x['x']
    u = model.u['u']
    lterm = (x.T @ Q @ x) + (u.T @ R @ u)
    mterm = x.T @ P @ x

    mpc.set_objective(mterm=mterm, lterm=lterm)

    max_x = np.array([[0.15], [0.15]])

    mpc.bounds['lower', '_x', 'x'] = -np.array([[-10.0], [-2.0]])
    mpc.bounds['upper', '_x', 'x'] = np.array([[2.0], [2.0]])

    mpc.bounds['lower', '_u', 'u'] = np.array([-1.0])
    mpc.bounds['upper', '_u', 'u'] = np.array([1.0])

    mpc.setup()

    estimator = do_mpc.estimator.StateFeedback(model)

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step=0.1)
    simulator.setup()

    # Seed
    np.random.seed(99)

    # Initial state
    e = np.ones([model.n_x,1])
    x0 = np.random.uniform(-3*e,3*e) # Values between +3 and +3 for all states
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    
    for k in range(50):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    tr = 5



    
    return None


class Polyhedron:
    def __init__(self, A=None, b=None, V=None):
        if V is not None:
            # V-representation: convex hull of vertices
            V = np.asarray(V, dtype=float)
            gen_mat = cdd.Matrix(
                np.hstack([np.ones((V.shape[0], 1)), V]).tolist(),
                number_type="float"
            )
            gen_mat.rep_type = cdd.RepType.GENERATOR
            self._poly = cdd.Polyhedron(gen_mat)

        elif A is not None and b is not None:
            # H-representation: {x | A x <= b}
            H_rows = [[b[i]] + (-A[i]).tolist() for i in range(len(b))]
            H_mat = cdd.Matrix(H_rows, number_type="float")
            H_mat.rep_type = cdd.RepType.INEQUALITY
            self._poly = cdd.Polyhedron(H_mat)

        else:
            raise ValueError("Must provide either (A,b) or V")
    
    # --- API methods ---
    def contains(self, x, tol=1e-9):
        """Check if point x is inside polyhedron."""
        H = self._poly.get_inequalities()
        H_np = np.array([list(row) for row in H], dtype=float)
        b = H_np[:, 0]
        A = -H_np[:, 1:]
        x = np.asarray(x, dtype=float)
        return np.all(A @ x <= b + tol)

    def vertices(self):
        """Return vertices (V-rep)."""
        gens = self._poly.get_generators()
        V = np.array([row[1:] for row in gens if row[0] == 1], dtype=float)
        return V

    def Ab(self):
        """Return (A,b) in Ax <= b form."""
        H = self._poly.get_inequalities()
        H_np = np.array([list(row) for row in H], dtype=float)
        b = H_np[:, 0]
        A = -H_np[:, 1:]
        return A, b
    

def sample_from_disturbance(W_polyhedron, n_samples=1):
    """Sample uniformly from the disturbance polytope."""
    vertices = W_polyhedron.vertices()
    
    # Method 1: Rejection sampling for small polytopes
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    samples = []
    while len(samples) < n_samples:
        candidate = np.random.uniform(min_coords, max_coords)
        if W_polyhedron.contains(candidate):
            samples.append(candidate)
    
    return np.array(samples) if n_samples > 1 else samples[0]

if __name__ == "__main__":
    main()
