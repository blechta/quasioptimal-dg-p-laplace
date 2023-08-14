import firedrake as fd

import numpy as np

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

        # FIXME: This has to be overloaded for a mixed method
        self.bcs = problem.bcs(Z)  # This is overloaded in DGSolver
        self.bcs = fd.solving._extract_bcs(self.bcs)

        #Obtain parameters from the constitutive relation and make sure they are Constants
        self.const_rel_params = {}
        for param_str, param in self.problem.const_rel_params.items():
            setattr(self, param_str, param)
            if not isinstance(getattr(self,param_str), fd.Constant):
                setattr(self, param_str, fd.Constant(getattr(self, param_str)))
            self.const_rel_params[param_str] = getattr(self, param_str)

        # Make sure "p" and "delta" are defined
        if not("p" in list(self.problem.const_rel_params.keys())): self.p = fd.Constant(2.0)
        if not("delta" in list(self.problem.const_rel_params.keys())): self.delta = fd.Constant(0.0)


    def solve(self, continuation_params):
        """ Rudimentary implementation of Newton + continuation
        'continuation_params' is a dictionary of the form {'param': param_list}
        specifying a list for each parameter over which we iterate"""

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

                # Run SNES
                self.nonlinear_variational_solver.solve()

            counter = 1


    @fd.utils.cached_property
    def nonlinear_variational_solver(self):

        if self.smoothing:
            F = self.lhs(self.z, self.z_)
            op = SmoothingOpVeeserZanotti(self.Z)
            def post_function_callback(_, residual):
                rhs = op.apply(self.problem.rhs)
                for bc in self.bcs:
                    bc.zero(rhs)
                with rhs.dat.vec_ro as v:
                    residual.axpy(-1, v)
        else:
            F = self.lhs(self.z, self.z_) - self.problem.rhs(self.z_)
            post_function_callback = None

        problem = fd.NonlinearVariationalProblem(F, self.z, bcs=self.bcs, J=self.get_jacobian())
        solver = fd.NonlinearVariationalSolver(problem,
                                               post_function_callback=post_function_callback,
                                               solver_parameters=self.get_parameters())
        return solver


    def get_jacobian(self):
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

    def natural_F(self, w_1, w_2=None, conjugate=False):
        if conjugate: # Computes the natural distance with p', delta^{p-1}
            p_ = float(self.p/(self.p-1.))
            delta_ = float(self.delta**(self.p - 1.))
        else:
            p_ = float(self.p)
            delta_ = float(self.delta)

        F_ = (delta_ + fd.inner(w_1, w_1)**(1/2.))**(0.5*(p_-2)) * w_1
        if w_2 is not None: F_ -= (delta_ + fd.inner(w_2, w_2)**(1/2.))**(0.5*(p_-2)) * w_2
        return (fd.assemble(fd.inner(F_, F_) * fd.dx))**0.5

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
        params = {"snes_monitor": None,
                  "snes_converged_reason": None,
                  #"snes_max_it": 20,
                  "snes_max_it": 100,
                  "snes_atol": 1e-8,
                  "snes_rtol": 1e-6,
                  "snes_dtol": 1e10,
                  "snes_type": "newtonls",
                  #"snes_linesearch_type": "none",
                  "snes_linesearch_type": "nleqerr",
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "ksp_converged_reason": None,
                  "ksp_monitor_true_residual": None,
                  "pc_factor_mat_solver_type": "mumps",
                  #"mat_mumps_icntl_14": 5000,
                  #"mat_mumps_icntl_24" : 1,
                  #"mat_mumps_cntl_1": 0.001,
                  #"mat_mumps_cntl_3": 0.0001,
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

    def __init__(self, problem, nref=1, solver_type="lu", k=1, smoothing=False, penalty_form="const_rel", no_shift=True):
        super().__init__(problem, nref=nref, solver_type=solver_type, k=k, smoothing=smoothing)
        self.penalty_form = penalty_form
        self.no_shift = no_shift
        assert penalty_form in ["quadratic", "p-d", "plaw", "const_rel"], "I don't know that form of the penalty..."

    def function_space(self, mesh, k):
        return fd.FunctionSpace(mesh, "CR", k)

    def ip_penalty_jump(self, h_factor, vec, form="const_rel"):
        """ Define the nonlinear part in penalty term using the constitutive relation or just using the Lp norm"""
        assert form in ["const_rel", "p-d", "plaw", "quadratic"], "That is not a valid form for the penalisation term"
        U_jmp = h_factor * vec
        if self.formulation_Su or (form == "p-d"):
            p = self.problem.const_rel_params.get("p", 2.0) # Get power-law exponent
            delta = self.problem.const_rel_params.get("delta", 0.0001) # Get delta
            jmp_penalty = (delta + fd.inner(U_jmp, U_jmp)) ** (0.5*p- 1) * U_jmp
        elif form == "const_rel":
            jmp_penalty = self.problem.const_rel(U_jmp)
        if form == "plaw":
            p = self.problem.const_rel_params.get("p", 2.0) # Get power-law exponent
            K_ = self.problem.const_rel_params.get("K", 1.0) # Get consistency index
            jmp_penalty = K_ * fd.inner(U_jmp, U_jmp) ** ((p-2.)/2.) * U_jmp
        elif form == "quadratic":
            K_ = self.problem.const_rel_params.get("K", 1.0) # Get consistency index
            jmp_penalty = K_ * U_jmp

        return jmp_penalty

    def modular(self, z):
        alpha = 10. * self.k**2
        n = fd.FacetNormal(self.Z.ufl_domain())
        h = fd.CellDiameter(self.Z.ufl_domain())
        self.penalty_form = "const_rel"
        U_jmp = 2. * fd.avg(fd.outer(z,n))
        U_jmp_bdry = fd.outer(z, n)
        jmp_penalty = self.ip_penalty_jump(1./fd.avg(h), U_jmp, form=self.penalty_form)
        jmp_penalty_bdry = self.ip_penalty_jump(1./h, U_jmp_bdry, form=self.penalty_form)
        jumps = fd.assemble(alpha * fd.inner(jmp_penalty, 2*fd.avg(fd.outer(z, n))) * fd.dS)
        jumps += fd.assemble(alpha * fd.inner(jmp_penalty_bdry, fd.outer(z,n)) * fd.ds)
        power = 2.0 if (self.penalty_form == "quadratic") else float(self.p)
        return (jumps)**(1./power)


class DGSolver(CrouzeixRaviartSolver):

    def __init__(self, problem, nref=1, solver_type="lu", k=1, smoothing=False, penalty_form="const_rel", no_shift=True):
        super().__init__(problem, nref=nref, solver_type=solver_type, k=k, smoothing=smoothing, penalty_form=penalty_form, no_shift=no_shift)
        self.bcs = ()

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

    def get_jacobian(self):
        if self.no_shift:# or (float(self.p) <= 2.0):
            J0 = fd.derivative(self.lhs(self.z, self.z_), self.z)
        else:
            # Define functions and test functions
            z_ = fd.TestFunction(self.Z)
            fields = self.split_variables(self.z, z_)
            u = fields["u"]
            v = fields["v"]
            S = fields.get("S")
            T = fields.get("T")
            # Also a trial function
            w = fd.TrialFunction(self.Z)
            # ==================== TEST =======================
            fields_ = self.split_variables(w, z_)
            u0 = fields_["u"]
            D0 = fd.grad(u0)
            # ==================== TEST =======================

            # Compute the max shift
            z_current = self.z.copy(deepcopy=True)
            if self.formulation_u:
                u_current = z_current
            elif self.formulation_Su:
                _, u_current = z_current.subfunctions
            else:
                raise(NotImplementedError)
            gradu_current_squared = fd.project(fd.inner(fd.grad(u_current), fd.grad(u_current)), fd.FunctionSpace(self.z.ufl_domain(), "DG", 2*(self.k-1)))
            max_shift = np.max(np.sqrt(np.absolute(gradu_current_squared.vector().get_local())))

            # For the DG terms
            alpha = 10. * self.k**2
            n = fd.FacetNormal(self.Z.ufl_domain())
            h = fd.CellDiameter(self.Z.ufl_domain())
            U_jmp = (2./fd.avg(h)) * fd.avg(fd.outer(u,n))
            U_jmp_bdry = (1./h) * fd.outer(u, n)
            delta = self.problem.const_rel_params.get("delta", 0.0001) # Get delta
            jmp_penalty = (delta + max_shift + fd.inner(U_jmp, U_jmp)) ** (0.5*self.p- 1) * U_jmp # We include the shift here!
            jmp_penalty_bdry = (delta + max_shift + fd.inner(U_jmp_bdry, U_jmp_bdry)) ** (0.5*self.p- 1) * U_jmp_bdry # We include the shift here!

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
                # ======================== TEST (write the Newton jacobian explicitly) ======================00
#                max_shift = 0.
                D = fd.grad(u)
                U_jmp0 = (2./fd.avg(h)) * fd.avg(fd.outer(u0,n))
                U_jmp_bdry0 = (1./h) * fd.outer(u0, n)
                jmp_penalty0 = (delta + max_shift + fd.inner(U_jmp, U_jmp)) ** (0.5*self.p- 1) * U_jmp0 # We include the shift here!
                jmp_penalty0 += (self.p - 2) * (delta + max_shift + fd.inner(U_jmp, U_jmp)) ** (0.5*self.p- 2) * fd.dot(U_jmp0,U_jmp) * U_jmp # We include the shift here!
                jmp_penalty_bdry0 = (delta + max_shift + fd.inner(U_jmp_bdry, U_jmp_bdry)) ** (0.5*self.p- 1) * U_jmp_bdry0 # We include the shift here!
                jmp_penalty_bdry0 += (delta + max_shift + fd.inner(U_jmp_bdry, U_jmp_bdry)) ** (0.5*self.p- 2) * fd.dot(U_jmp_bdry0, U_jmp_bdry) * U_jmp_bdry # We include the shift here!
                J0 = (
                    fd.inner((self.delta + fd.inner(D,D))**(0.5*self.p - 1) * D0, fd.grad(v)) * fd.dx
                    + (self.p - 2) * (self.delta + fd.inner(D,D))**(0.5*self.p - 2) * fd.dot(D0, fd.grad(u)) * fd.dot(fd.grad(v), fd.grad(u)) * fd.dx
                    - fd.inner(fd.avg((self.delta + fd.inner(D,D))**(0.5*self.p - 1) * D0), 2*fd.avg(fd.outer(v, n))) * fd.dS
                    - (self.p - 2) * fd.inner(fd.avg((self.delta + fd.inner(D,D))**(0.5*self.p - 2) * fd.dot(D0, fd.grad(u))*fd.grad(u)), 2*fd.avg(fd.outer(v, n))) * fd.dS
                    - fd.inner((self.delta + fd.inner(D,D))**(0.5*self.p - 1) * D0, fd.outer(v, n)) * fd.ds
                    - (self.p - 2) * fd.inner((self.delta + fd.inner(D,D))**(0.5*self.p - 2) * fd.dot(D0, fd.grad(u))*fd.grad(u), fd.outer(v, n)) * fd.ds
                    + alpha * fd.inner(jmp_penalty0, 2*fd.avg(fd.outer(v, n))) * fd.dS
                    + alpha * fd.inner(jmp_penalty_bdry0, fd.outer(v,n)) * fd.ds
                )
                # ======================== TEST (write the Newton jacobian explicitly) ======================00
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

#            J0 = fd.derivative(F, self.z, w)

            return J0

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
