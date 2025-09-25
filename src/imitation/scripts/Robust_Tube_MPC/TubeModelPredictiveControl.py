import casadi as ca

class TubeModelPredictiveControl():

    def __init__(self, sys, optcon, Xc, Uc, Xc_robust, Xmpi_robust, N, solution_cache):

        self.sys = sys                  # linear system with disturbance
        self.optcon = optcon            # optimal control solver object
        self.Xc = Xc                    # state constraint
        self.Uc = Uc                    # input constraint
        self.Xc_robust = Xc_robust      # Xc-Z (pontryagin diff.)
        self.Xmpi_robust = Xmpi_robust  # MPI set 
        self.N = N                      # horizon
        self.solution_cache = solution_cache
