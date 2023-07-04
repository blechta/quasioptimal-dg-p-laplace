import firedrake as fd

from smoothing import SmoothingOpVeeserZanotti

class NonlinearEllipticProblem(object):
    def __init__(self, **const_rel_params):
        # We will work with primal and mixed formulations
        self.formulation = "u"
        # Define constitutive parameters; e.g. the power-law exponent
        self.const_rel_params = const_rel_params
        for param in const_rel_params.keys():
            setattr(self, param, const_rel_params[param])

    def mesh(self): raise NotImplementedError

    def mesh_hierarchy(self, nref):
        base = self.mesh()
        return fd.MeshHierarchy(base, nref)

    def bcs(self, Z):
        #For now we only do homogeneous BCs
        return fd.DirichletBC(Z, fd.Constant(0.), "on_boundary")

    def const_rel(self, *args): raise NotImplementedError

    def rhs(self, Z): raise NotImplementedError


class NonlinearEllipticProblem_Su(NonlinearEllipticProblem):
    """Problem Class for the mixed formulation"""
    def __init__(self, **const_rel_params):
        super().__init__(**const_rel_params)
        self.formulation = "S-u"


class NonlinearEllipticSolver(object):

    def function_space(self, mesh, k=1): raise NotImplementedError

    def __init__(self, problem, nref=1, solver_type="lu", k=1, smoothing=False):

        self.problem = problem
        self.nref = nref
        self.solver_type = solver_type
        self.smoothing = smoothing
        self.formulation_u = (self.problem.formulation == "u")
        self.formulation_Su = (self.problem.formulation == "S-u")

        mh = problem.mesh_hierarchy(self.nref)
        self.mh = mh
        mesh = mh[-1]

        Z = self.function_space(mesh, k)
        self.Z = Z
        print("Number of dofs: %s" % (self.Z.dim()))

        z = fd.Function(Z, name="solution")
        self.z = z

        bcs = problem.bcs(Z)
        self.bcs = bcs


    def solve(self, maxiter=20, atol=1e-8, rtol=1e-6):
        """ Rudimentary implementation of Newton """

        # The update
        deltaz = fd.Function(self.Z)

        # Do the actual loop
        for n in range(maxiter+1):
            # Raise an error if the maximum number of iterations was already reached
            if (n == maxiter): raise RuntimeError("The Newton solver did not converge after %i iterations"%maxiter)

            # Assemble the linearised system around u
            A, b = self.assemble_system()

            # Cast the matrix to a sparse format and use a sparse solver for
            # the linear system. This is vastly faster than the dense
            # alternative.
            print("Current Newton iteration: %i"%n)
#            fd.solve(A, deltaz, b, bcs=self.bcs, solver_parameters=self.get_parameters())
            fd.solve(A, deltaz, b, solver_parameters=self.get_parameters())

            # Update the solution and check for convergence
            self.z.assign(self.z + deltaz)

            # Relative tolerance: compare relative to the current guess 
#            p = self.problem.const_rel_params.get("p", 2.0) # Get power-law exponent
            p = 2 #FIXME: Compute with the Lp norm
            p = str(p)
            print("----- Computing with the %s norm", ("L"+p))
            relerror = fd.norm(deltaz, norm_type="l"+p) / fd.norm(self.z, norm_type="l"+p)
            print("--------- Relative error = ", relerror)
            if (relerror < rtol):
                print("Correction satisfied the relative tolerance!")
                break
            # Absolute tolerance: distance of deltau to zero
            abserror = fd.norm(deltaz, norm_type="l"+p)
            print("--------- Absolute error = ", abserror)
            if (abserror < atol):
                print("Correction satisfied the absolute tolerance!")
                break
            # TODO: Print also residual?


    def assemble_system(self):
        J = self.get_jacobian()
        A = fd.assemble(J, bcs=self.bcs)

        if self.smoothing:
            op = SmoothingOpVeeserZanotti
#            def rhs_(test_f): return self.problem.rhs()
            b = op.apply(lambda test_f : self.problem.rhs(test_f.function_space()))
#            b = op.apply(self.problem.rhs)
        else:
            b = fd.assemble(self.problem.rhs(self.Z))

        return A, b

    def get_jacobian(self):
        # Later we will need something other than Newton
        J0 = fd.derivative(self.lhs(), self.z)
        return J0


    def split_variables(self, z):
        Z = self.Z
        fields = {}
        if self.problem.formulation == "u":
            v = fd.TestFunction(Z)
            fields["u"] = z
        elif self.problem.formulation == "S-u":
            (S, u) = fd.split(z)
            (T, v) = fd.split(fd.TestFunction(Z))
            fields["u"] = u
            fields["S"] = S
            fields["T"] = T
        else:
            raise NotImplementedError
        fields["v"] = v
        return fields


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

    def lhs(self): raise NotImplementedError

class ConformingSolver(NonlinearEllipticSolver):

    def function_space(self, mesh, k):
        return fd.FunctionSpace(mesh, "CG", k)

    def lhs(self):

        # Define functions and test functions
        fields = self.split_variables(self.z)
        u = fields["u"]
        v = fields["v"]
        S = fields.get("S")
        T = fields.get("T")

        if self.formulation_u:
            G = self.problem.const_rel(fd.grad(u))
        elif self.formulation_Su:
            G = self.problem.const_rel(S)
        else:
            raise NotImplementedError

        if self.formulation_u:
            F = fd.inner(G, fd.grad(v)) * fd.dx
        elif self.formulation_Su:
            F = (
                fd.inner(S, fd.grad(v)) * fd.dx
                + fd.inner(T, fd.grad(u)) * fd.dx
                - fd.inner(G, T) * fd.dx
            )
        else:
            raise NotImplementedError
        return F


#class CrouzeixRaviartSolver(ConformingSolver): raise NotImplementedError

#class DGSolver(NonlinearEllipticSolver): raise NotImplementedError
