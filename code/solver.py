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

    def rhs(self, z_): raise NotImplementedError


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
        self.k = k

        mh = problem.mesh_hierarchy(self.nref)
        self.mh = mh
        mesh = mh[-1]

        Z = self.function_space(mesh, k)
        self.Z = Z
        print("Number of dofs: %s" % (self.Z.dim()))

        z = fd.Function(Z, name="solution")
        self.z = z
        self.z_ = fd.TestFunction(Z)

        bcs = problem.bcs(Z)
        self.bcs = bcs

        #Obtain parameters from the constitutive relation and make sure they are Constants
        self.const_rel_params = {}
        for param_str, param in self.problem.const_rel_params.items():
            setattr(self, param_str, param)
            if not isinstance(getattr(self,param_str), fd.Constant):
                setattr(self, param_str, fd.Constant(getattr(self, param_str)))
            self.const_rel_params[param_str] = getattr(self, param_str)


    def solve(self, continuation_params, maxiter=20, atol=1e-8, rtol=1e-6):
        """ Rudimentary implementation of Newton + continuation
        'continuation_params' is a dictionary of the form {'param': param_list}
        specifying a list for each parameter over which we iterate"""

        # The Newton update
        deltaz = fd.Function(self.Z)

        # We will do continuation in the constitutive parameters
        #Set all the initial parameters
        for param_str, param_list in continuation_params.items():
            getattr(self, param_str).assign(param_list[0])

        # Continuation loop
        counter = 0 # Otherwise it does the last solve twice
        for param_str in continuation_params.keys():
            for param in continuation_params[param_str][counter:]:
                getattr(self, param_str).assign(param)

                output_info = "Solving for "
                for param_ in continuation_params.keys():
                    output_info += param_
                    output_info += " = %.8f, "%float(getattr(self, param_))
                    fd.warning(fd.RED % (output_info))

                # Newton loop
                for n in range(maxiter+1):
                    # Raise an error if the maximum number of iterations was already reached
                    if (n == maxiter): raise RuntimeError("The Newton solver did not converge after %i iterations"%maxiter)

                    # Assemble the linearised system around u
                    A, b = self.assemble_system()

                    # Cast the matrix to a sparse format and use a sparse solver for
                    # the linear system. This is vastly faster than the dense
                    # alternative.
                    fd.warning(fd.BLUE % "Current Newton iteration: %i"%n)
                    fd.solve(A, deltaz, b, solver_parameters=self.get_parameters())
        #====================== TEST ========================================
        #            a_ = fd.inner(fd.grad(fd.TrialFunction(self.Z)), fd.grad(fd.TestFunction(self.Z))) * fd.dx
        #            L_ = -fd.inner(fd.grad(self.z), fd.grad(fd.TestFunction(self.Z))) * fd.dx
        #            L_ += self.problem.rhs(self.Z)
        #            fd.solve(a_ == L_, deltaz, bcs=self.bcs)
        #====================== TEST ========================================

                    # Update the solution and check for convergence
                    self.z.assign(self.z + deltaz)

                    # Print residual (not used for convergence criteria)
                    F = self.lhs(self.z, self.z_)
                    F -= self.problem.rhs(self.z_)
                    F = fd.assemble(F)
                    fd.warning(fd.BLUE % "--------- Residual norm (in the L2 norm)  = %.14e" % fd.norm(F))

                    # Relative tolerance: compare relative to the current guess 
                    p_ = float(self.problem.const_rel_params.get("p", 2.0)) # Get power-law exponent
                    norm_type = "W^{1, %s}"%str(p_) if self.formulation_u else "L^{%s} x W^{1,%s}"%(str(p_/(p_-1.)), str(p_))
                    relerror = self.W1pnorm(deltaz, p_) / self.W1pnorm(self.z, p_)
                    fd.warning(fd.BLUE % "--------- Relative error (in the %s norm) = " % norm_type + str(relerror))
                    if (relerror < rtol):
                        fd.warning(fd.GREEN % "Converged due to relative tolerance!")
                        break
                    # Absolute tolerance: distance of deltau to zero
                    abserror = self.W1pnorm(deltaz, p_)
                    fd.warning(fd.BLUE % "--------- Absolute error (in the %s) = "%norm_type + str(abserror))
                    if (abserror < atol):
                        fd.warning(fd.GREEN % "Converged due to absolute tolerance!")
                        break

            counter = 1


    def assemble_system(self):
        J = self.get_jacobian()
        A = fd.assemble(J, bcs=self.bcs)

        if self.smoothing:
            def newton_rhs(v):
                nrhs = -self.lhs(self.z, v)
                nrhs += self.problem.rhs(v)
                return nrhs
            op = SmoothingOpVeeserZanotti(self.Z)
#            b = op.apply(lambda test_f : -self.residual())                 # This doesn't work
            b = op.apply(newton_rhs)
        else:
            b = -self.residual()
            b = fd.assemble(b)

        return A, b

    def residual(self):
        return self.lhs(self.z, self.z_) - self.problem.rhs(self.z_)


    def get_jacobian(self):
        # Later we will need something other than Newton
        J0 = fd.derivative(self.lhs(self.z, self.z_), self.z)
        return J0


    def split_variables(self, z, z_):
        fields = {}
        if self.problem.formulation == "u":
            fields["u"] = z
            fields["v"] = z_
        elif self.problem.formulation == "S-u":
            (S, u) = fd.split(z)
            (T, v) = fd.split(z_)
            fields["u"] = u
            fields["v"] = v
            fields["S"] = S
            fields["T"] = T
        else:
            raise NotImplementedError
        return fields

    def W1pnorm(self, z, p):
        if self.formulation_u:
            return fd.assemble(fd.inner(fd.grad(z), fd.grad(z))**(p/2.) * fd.dx)**(1/p)
        elif self.formulation_Su:
            p_prime = p/(p-1.)
            (S, u) = z.split()
            S_norm = fd.assemble(fd.inner(S, S)**(p_prime/2.) * fd.dx)**(1/p_prime)
            u_norm = fd.assemble(fd.inner(fd.grad(u), fd.grad(u))**(p/2.) * fd.dx)**(1/p)
            return S_norm + u_norm

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

    def lhs(self, z, z_): raise NotImplementedError

class ConformingSolver(NonlinearEllipticSolver):

    def function_space(self, mesh, k):
        return fd.FunctionSpace(mesh, "CG", k)

    def lhs(self, z, z_):

        # Define functions and test functions
        fields = self.split_variables(z, z_)
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


class CrouzeixRaviartSolver(ConformingSolver):

    def function_space(self, mesh, k):
        return fd.FunctionSpace(mesh, "CR", k)

class DGSolver(NonlinearEllipticSolver):

    def __init__(self, problem, nref=1, solver_type="lu", k=1, smoothing=False, penalty_form="const_rel"):
        super().__init__(problem, nref=nref, solver_type=solver_type, k=k, smoothing=smoothing)
        self.penalty_form = penalty_form
        assert penalty_form in ["quadratic", "plaw", "const_rel"], "I don't know that form of the penalty..."

    def function_space(self, mesh, k):
        return fd.FunctionSpace(mesh, "DG", k)

    def lhs(self, z, z_):

        # Define functions and test functions
        fields = self.split_variables(z, z_)
        u = fields["u"]
        v = fields["v"]
        S = fields.get("S")
        T = fields.get("T")

        # For the DG terms
        alpha = 10. * self.k**2
        n = fd.FacetNormal(self.Z.ufl_domain())
        h = fd.CellDiameter(self.Z.ufl_domain())
        U_jmp = 2. * fd.avg(fd.outer(u,n))
        U_jmp_bdry = fd.outer(u, n)
        jmp_penalty = self.ip_penalty_jump(1./fd.avg(h), U_jmp, form=self.penalty_form)
        jmp_penalty_bdry = self.ip_penalty_jump(1./h, U_jmp_bdry, form=self.penalty_form)


        if self.formulation_u:
            G = self.problem.const_rel(fd.grad(u))
        elif self.formulation_Su:
            G = self.problem.const_rel(S)
        else:
            raise NotImplementedError

        if self.formulation_u:
            F = (
                fd.inner(G, fd.grad(v)) * fd.dx # TODO: Need a different formulation for LDG
                - fd.inner(fd.avg(G), 2*fd.avg(fd.outer(v, n))) * fd.dS
                - fd.inner(G, fd.outer(v, n)) * fd.ds
                + alpha * fd.inner(jmp_penalty, 2*fd.avg(fd.outer(v, n))) * fd.dS
                + alpha * fd.inner(jmp_penalty_bdry, fd.outer(v,n)) * fd.ds
            )
        elif self.formulation_Su:
            F = (
                -fd.inner(G, T) * fd.dx
                + fd.inner(fd.grad(u), T) * fd.dx
                - fd.inner(fd.avg(T), 2*fd.avg(fd.outer(u, n))) * fd.dS # Remove this one for IIDG
                - fd.inner(T, fd.outer(u, n)) * fd.ds # Remove this one for IIDG
                + fd.inner(S, fd.grad(v)) * fd.dx
                - fd.inner(fd.avg(S), 2*fd.avg(fd.outer(v, n))) * fd.dS
                - fd.inner(S, fd.outer(v, n)) * fd.ds
                + alpha * inner(jmp_penalty, 2*fd.avg(fd.outer(v, n))) * fd.dS
                + alpha * fd.inner(jmp_penalty_bdry, fd.outer(v,n)) * fd.ds
            )
        else:
            raise NotImplementedError
        return F

    def ip_penalty_jump(self, h_factor, vec, form="const_rel"):
        """ Define the nonlinear part in penalty term using the constitutive relation or just using the Lp norm"""
        assert form in ["const_rel", "plaw", "quadratic"], "That is not a valid form for the penalisation term"
        U_jmp = h_factor * vec
        if form == "const_rel" and self.formulation_u:
            jmp_penalty = self.problem.const_rel(U_jmp)
        if form == "plaw":
            p = self.problem.const_rel_params.get("p", 2.0) # Get power-law exponent
            K_ = self.problem.const_rel_params.get("K", 1.0) # Get consistency index
            jmp_penalty = K_ * fd.inner(U_jmp, U_jmp) ** ((p-2.)/2.) * U_jmp
        elif form == "quadratic":
            K_ = self.problem.const_rel_params.get("K", 1.0) # Get consistency index
            jmp_penalty = K_ * U_jmp
        return jmp_penalty


    def W1pnorm(self, z, p):

        # For the DG terms
        alpha = 10. * self.k**2
        n = fd.FacetNormal(self.Z.ufl_domain())
        h = fd.CellDiameter(self.Z.ufl_domain())
        if self.formulation_u:
            U_jmp = 2. * fd.avg(fd.outer(z,n))
            U_jmp_bdry = fd.outer(z, n)
            jmp_penalty = self.ip_penalty_jump(1./fd.avg(h), U_jmp, form=self.penalty_form)
            jmp_penalty_bdry = self.ip_penalty_jump(1./h, U_jmp_bdry, form=self.penalty_form)
            broken_W1p = fd.assemble(fd.inner(fd.grad(z), fd.grad(z))**(p/2.) * fd.dx)
            jumps = fd.assemble(alpha * fd.inner(jmp_penalty, 2*fd.avg(fd.outer(z, n))) * fd.dS)
            jumps += fd.assemble(alpha * fd.inner(jmp_penalty_bdry, fd.outer(z,n)) * fd.ds)
            return (broken_W1p + jumps)**(1/p)
        elif self.formulation_Su:
            (S, u) = z.split()
            U_jmp = 2. * fd.avg(fd.outer(u,n))
            U_jmp_bdry = fd.outer(u, n)
            jmp_penalty = self.ip_penalty_jump(1./fd.avg(h), U_jmp, form=self.penalty_form)
            jmp_penalty_bdry = self.ip_penalty_jump(1./h, U_jmp_bdry, form=self.penalty_form)
            p_prime = p/(p-1.)
            S_norm = fd.assemble(fd.inner(S, S)**(p_prime/2.) * fd.dx)**(1/p_prime)
            broken_W1p = fd.assemble(fd.inner(fd.grad(u), fd.grad(u))**(p/2.) * fd.dx)
            jumps = fd.assemble(alpha * fd.inner(jmp_penalty, 2*fd.avg(z)) *fd.dS)
            jumps += fd.assemble(alpha * fd.inner(jmp_penalty_bdry, z) * fd.ds)
            return S_norm + (broken_W1p + jumps)**(1/p)
