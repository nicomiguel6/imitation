import numpy as np
import sys
import casadi as ca
import do_mpc
import cdd
from scipy.linalg import solve_discrete_are
from scipy.spatial import ConvexHull
import pytope

def main():

    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    _x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

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
    W_polytope = pytope.Polytope(W_vertex)
    gen_mat = cdd.Matrix(
        np.hstack([np.ones((W_vertex.shape[0], 1), dtype=object), W_vertex]).tolist(),
        number_type='float'
    )
    gen_mat.rep_type = cdd.RepType.GENERATOR

    W = cdd.Polyhedron(gen_mat)
    W_polyhedron = Polyhedron(V = W_vertex)

    # Convert to H-rep (Ax <= b)
    H = W.get_inequalities()
    H_mat = np.array([list(row) for row in H], dtype=float)
    W_b, W_A = H_mat[:,0], -H_mat[:,1:]

    A_x, b_x = box_to_Ab(np.array([[-10.0], [-2.0]]), np.array([[2.0], [2.0]]))
    A_u, b_u = box_to_Ab(np.array([-1.0]), np.array([1.0]))

    # Compute mrpi set
    A_F, b_F = compute_mrpi_hrep(Ak, W_polytope.A, W_polytope.b)

    # Tighten
    A_x_t, b_x_t = minkowski_difference_Hrep(A_x, b_x, A_F, b_F)
    A_u_t, b_u_t = tighten_by_linear_image_Hrep(A_u, b_u, K, A_F, b_F)

    # compute robust MPI set
    Xc_robust = pytope.Polytope(A = A_x_t, b = b_x_t)
    Uc_robust = pytope.Polytope(A = A_u_t, b = b_u_t)

    print("State mRPI: ", Xc_robust.V)
    print("Control mRPI: ", Uc_robust.V)



    x_next = A@_x + B@_u

    model.set_rhs('x', x_next)

    model.set_expression(expr_name='cost', expr= _x.T @ Q @ _x)

    # Build the model
    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 10,
        't_step': 0.1,
        'state_discretization': 'discrete',
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mterm = model.aux['cost'] # terminal cost
    lterm = model.aux['cost'] # terminal cost
    # stage cost

    mpc.settings.set_linear_solver()

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(u=R) # input penalty



    # lower bounds of the states
    mpc.bounds['lower','_x','x'] = np.array([-b_x_t[0], -b_x_t[2]])

    # upper bounds of the states
    mpc.bounds['upper','_x','x'] = np.array([b_x_t[1], b_x_t[3]])

    # lower bounds of the input
    mpc.bounds['lower','_u','u'] = -np.array([b_u_t[0]])

    # upper bounds of the input
    mpc.bounds['upper','_u','u'] =  np.array([b_u_t[0]])


    print("A_x_t: ", str(A_x_t))
    print("b_x_t: ", str(b_x_t))
    print("-----------------------------------------------------------------")

    mpc.setup()


    estimator = do_mpc.estimator.StateFeedback(model)

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.1)
    simulator.setup()

    # Seed
    np.random.seed(99)

    # Initial state
    e = np.ones([model.n_x,1])
    x0 = np.array([-7,-2]) # Values between +3 and +3 for all states
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    for k in range(50):
        u0 = mpc.make_step(x0)
        x_nom0 = mpc.data.prediction(('_x', 'x'))[:,0]
        print("Predicted trajectory: ", mpc.data.prediction(('_x', 'x')))
        u_applied = u0 + K @ (x0.T.reshape(-1,) - x_nom0.reshape(-1,))
        y_next = simulator.make_step(u_applied)
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
    V = get_vertices(A, b)

    d = np.asarray(d)
    d = np.atleast_2d(d)      # shape (m, n_dirs)
    return np.max(V @ d, axis=0)

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

def compute_MPI_set(Ak, K, X, U):
    F, G, nc = convert_Poly2Mat(X, U)
    def Fpi(i):
        return (F + G @ K) @ np.linalg.matrix_power(Ak, i)
    def Xpi(i): 
        return pytope.Polytope(A = Fpi(i), b = np.ones((Fpi(i).shape[0], 1)))
    Xmpi = Xpi(0)
    i = 0
    while 1:
        i += 1
        Xpi_i = Xpi(i)
        Xmpi_tmp = pytope.polytope.intersection(Xmpi, Xpi_i)
        if Xmpi_tmp.__eq__(Xmpi):
            break
        else:
            Xmpi = Xmpi_tmp


    return Xmpi

def convert_Poly2Mat(X, U):
    """convert Polyhedron to matrix of inequalities

    Args:
        X (_type_): _description_
        U (_type_): _description_
    """

    # extract inequality matrix for X
    def poly2ineq_norm(poly):
        return poly.A/np.tile(poly.b, (1, np.shape(poly.A)[1]))
    
    F_tmp = poly2ineq_norm(X)
    G_tmp = poly2ineq_norm(U)
    
    X_dim = np.linalg.matrix_rank(X.V - X.V[0])
    U_dim = np.linalg.matrix_rank(U.V - U.V[0])
    # check for empty matrices
    if np.prod(np.shape(F_tmp)) == 0:
        F_tmp = np.empty((0,X_dim))
    if np.prod(np.shape(G_tmp)) == 0:
        G_tmp = np.empty((0,U_dim))

    F = np.vstack([F_tmp, np.zeros((G_tmp.shape[0], X_dim))])
    G = np.vstack([np.zeros((F_tmp.shape[0], U_dim)), G_tmp])
    nc = np.shape(F)[0]


    return F, G, nc



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

def tighten_by_linear_image_Vrep(A_u, b_u, K, A_F, b_F):
    """ 
    Tighten input constraints U ⊖ K F in V-rep.
    A_u, b_u : original input constraints
    K        : feedback gain matrix
    A_F, b_F : H-rep of disturbance/error set F
    """

    # Begin by converting [A_u, b_u] and [A_F, b_F] into vertices
    Uc = get_vertices(A_u, b_u)
    Fc = get_vertices(A_F, b_F)

    # multiply mRPI vertices by K
    Fc = Fc @ np.diag(K)

    # Since axis aligned...?
    # Uc_robust = 

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