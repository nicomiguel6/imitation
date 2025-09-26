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
    K = -np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # LQR gain

    # closed loop
    Ak = A + B@K

    # Set up disturbance with cdd
    W_vertex = np.array([[0.15, 0.15],[0.15, -0.15],[-0.15,-0.15],[-0.15,0.15]], dtype=object)
    gen_mat = cdd.Matrix(
        np.hstack([np.ones((W_vertex.shape[0], 1), dtype=object), W_vertex]).tolist(),
        number_type='fraction'
    )
    gen_mat.rep_type = cdd.RepType.GENERATOR

    W = cdd.Polyhedron(gen_mat)
    W_polyhedron = Polyhedron(V = W_vertex)

    # Convert to H-rep (Ax <= b)
    H = W_poly.get_inequalities()
    H_mat = np.array([list(row) for row in H], dtype=float)
    W_b, W_A = H_mat[:,0], -H_mat[:,1:]

    # Compute mrpi set
    V_F = compute_mrpi(Ak, W_A, W_b)

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



    mpc.bounds['lower', '_x', 'x'] = np.array([[-10.0], [-2.0]])
    mpc.bounds['upper', '_x', 'x'] = np.array([[2.0], [2.0]])

    mpc.bounds['lower', '_u', 'u'] = np.array([-1.0])
    mpc.bounds['upper', '_u', 'u'] = np.array([1.0])

    A_x, b_x = box_to_Ab(np.array([[-10.0], [-2.0]]), np.array([[2.0], [2.0]]))
    A_u, b_u = box_to_Ab(np.array([-1.0]), np.array([1.0]))


    tvp_template = mpc.get_tvp_template()

    def tube_constraints(t_now):
        tvp_template['_tvp', 'x_upper'] = 



    mpc.setup()

    estimator = do_mpc.estimator.StateFeedback(model)

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step=0.1)
    simulator.setup()

    # Seed
    np.random.seed(99)

    # Initial state
    e = np.ones([model.n_x,1])
    x0 = np.random.uniform(-3*e,3*e) # Values between -3 and +3 for all states
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    
    for k in range(50):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    from matplotlib import rcParams
    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18

    import matplotlib.pyplot as plt
    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16,9))
    graphics.plot_results()
    graphics.reset_axes()
    plt.show()




    
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

def support_function(A, b, d):
    # cdd wants [b | A] with inequalities in form b + A x >= 0
    mat = cdd.Matrix(
        np.hstack([b.reshape(-1,1), -A]).tolist(),
        number_type="fraction"
    )
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)

    verts = poly.get_generators()
    V = np.array([row[1:] for row in verts if row[0] == 1], dtype=float)

    return np.max(V @ d)

def minkowski_sum(V1, V2):
    """
    Approximate Minkowski sum of two polytopes in V-rep:
    P = conv(V1), Q = conv(V2)
    return conv(V1 + V2)
    """
    V_sum = np.array([v1 + v2 for v1 in V1 for v2 in V2])
    # Convex hull via cdd
    mat = cdd.Matrix(
        np.hstack([np.ones((V_sum.shape[0],1)), V_sum]).tolist(),
        number_type="fraction"
    )
    mat.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(mat)
    verts = P.get_generators()

    return np.array([row[1:] for row in verts if row[0] == 1], dtype=float)

def minkowski_difference(Ax, bx, V_F):
    """
    Compute X ⊖ F where
      X = {x | Ax x <= bx}
      F = conv(V_F) given in V-rep (vertices)
    Returns (A_diff, b_diff) for tightened polytope
    """
    b_new = []
    for i in range(Ax.shape[0]):
        a = Ax[i,:]
        # support of F in direction a
        hF = np.max(V_F @ a)
        b_new.append(bx[i] - hF)

    return Ax, np.array(b_new)

def compute_mrpi(Ak, W_A, W_b, epsilon=1e-5, max_iter=50):
    """
    Compute outer approximation of mRPI set for e^+ = A e + w, w in W.
    Inputs:
        Ak : closed-loop matrix
        W_A, W_b : polytope in H-rep (W = {x | W_A x <= W_b})
        epsilon : tolerance
    Returns:
        V_F : vertices of approximate mRPI set
    """
    nx = Ak.shape[0]
    s = 0
    alpha, Ms = 1000, 1000

    # Step 1: find s such that alpha small enough
    while alpha > epsilon/(epsilon + Ms) and s < max_iter:
        s += 1
        # Compute alpha
        dirs = (Ak**s) @ W_A.T  # directions = A^s * facet normals
        alpha = np.max([support_function(W_A, W_b, d) / bi
                        for d, bi in zip(dirs, W_b)])

        # Update Ms
        mss = []
        for i in range(1, s+1):
            d_pos = np.linalg.matrix_power(Ak, i)
            d_neg = -np.linalg.matrix_power(Ak, i)
            mss.append(support_function(W_A, W_b, d_pos @ np.ones(nx)))
            mss.append(support_function(W_A, W_b, d_neg @ np.ones(nx)))
        Ms = max(mss) if mss else Ms

    # Step 2: build finite Minkowski sum
    # Start from W in V-rep
    mat_W = cdd.Matrix(
        np.hstack([W_b.reshape(-1,1), -W_A]).tolist(),
        number_type="fraction"
    )
    mat_W.rep_type = cdd.RepType.INEQUALITY
    P_W = cdd.Polyhedron(mat_W)
    V_W = np.array([row[1:] for row in P_W.get_generators() if row[0] == 1], dtype=float)

    V_F = V_W.copy()
    for i in range(1, s):
        V_i = (np.linalg.matrix_power(Ak, i) @ V_W.T).T
        V_F = minkowski_sum(V_F, V_i)

    # Step 3: scale by 1/(1-alpha)
    V_F = (1.0/(1-alpha)) * V_F

    return V_F


def tighten_by_linear_image(AU, bU, K, V_F):
    """
    Return H-rep of U ⊖ (K F), where:
      U = {u | AU u <= bU},
      F = conv(V_F) (vertices, shape [nV, n_x]),
      K (shape [n_u, n_x]).
    """
    b_new = []
    KT = K.T  # shape (n_x, n_u)
    for i in range(AU.shape[0]):
        a = AU[i, :]                    # (n_u,)
        a_in_x_space = KT @ a           # (n_x,)
        hKF = np.max(V_F @ a_in_x_space)  # support of F in direction K^T a
        b_new.append(bU[i] - hKF)

    return AU, np.asarray(b_new)

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

if __name__ == "__main__":
    main()
