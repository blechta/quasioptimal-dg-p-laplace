import firedrake as fd

from smoothing import SmoothingOpVeeserZanotti

class NonlinearEllipticProblem(object):

    def mesh(self): raise NotImplementedError

    def mesh_hierarchy(self, nref):
        base = self.mesh()
        return fd.MeshHierarchy(base, nref)

    def bcs(self, Z):
        #For now we only do homogeneous BCs
        return fd.DirichletBC(Z.ufl_domain(), fd.Constant(0.), "on_boundary")

    def lhs(self, Z): raise NotImplementedError

    def rhs(self, Z): return None

    def jacobian(self, Z): raise NotImplementedError

class NonlinearEllipticSolver(object):

    def function_space(self, mesh, k=1):
        raise NotImplementedError

    def residual(self): raise NotImplementedError

    def __init__(self, problem, nref=1, solver_type="lu", k=1, smoothing=False):

        self.problem = problem
        self.nref = nref
        self.solver_type = solver_type
        self.smoothing = smoothing

        mh = problem.mesh_hierarchy(self.nref)
        self.mh = mh
        mesh = mh[-1]

        Z = self.function_space(mesh, k)
        self.Z = Z
        print("Number of dofs: %s" % (self.Z.dim()))

        u = fd.Function(Z, name="solution")
        v = fd.TestFunction(Z)
        self.u = u

        bcs = problem.bcs(Z)

        F = self.residual()

        if rhs is not None:
            if smoothing:
                op = SmoothingOpVeeserZanotti
                op.apply(rhs)
            F -= fd.inner(rhs, v) * fd.dx
        self.F = F
        self.J = self.get_jacobian()


        problem = fd.NonlinearVariationalProblem(F, u, bcs=bcs, J=self.J)
        self.params = self.get_parameters()
        self.solver = fd.NonlinearVariationalSolver(problem, solver_parameters=self.params)

    def get_jacobian(self):
        # Later we will need something other than Newon
        J0 = fd.derivative(self.F, self.u)
        return J0

    def solve(self):
        self.solver.solve()
        return self.u

    def get_parameters(self):
        #LU for now I guess...
        params = {"snes_monnitor": None,
                  "snes_converged_reason": None,
                  "snes_max_it": 120,
                  "snes_atol": 1e-8,
                  "snes_rtol": 1e-7,
                  "snes_dtol": 1e10,
                  "snes_linesearch_type": "nleqerr",
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "ksp_converged_reason": None,
                  "ksp_monitor_true_residual": None,
                  "pc_factor_mat_solver_type": "mumps",
                  "mat_mumps_icntl_14": 5000,
                  "mat_mumps_icntl_24" : 1,
                  "mat_mumps_cntl_1": 0.001,
                  "mat_mumps_cntl_3": 0.0001,
                  }
        return params

def ConformingSolver(NonlinearEllipticSolver): raise NotImplementedError

def CrouzeixRaviartSolver(ConformingSolver): raise NotImplementedError

def DGSolver(NonlinearEllipticSolver): raise NotImplementedError
