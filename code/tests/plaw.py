import firedrake as fd

import argparse
import sys
sys.path.append('..')

from solver import NonlinearEllipticProblem, ConformingSolver, CrouzeixRaviartSolver, DGSolver

class PowerLawTest(NonlinearEllipticProblem):
    def __init__(self, baseN, p, K, diagonal=None):
        super().__init__(p=p, K=K)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self):
        return fd.UnitSquareMesh(self.baseN, self.baseN, diagonal=self.diagonal)

    def const_rel(self, D):
        return self.K * fd.inner(D, D) ** ((float(self.p) - 2)/2.) * D

    def exact_solution(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        return fd.sin(4*fd.pi*x) * y**2 * (1.-y)**2

    def rhs(self, v):
        sols = self.exact_solution(v.function_space())
#        S = self.const_rel(fd.grad(sols))
#        L = - fd.div(S) * v * fd.dx
#        L = fd.inner(S, fd.grad(v)) * fd.dx # Should this work for DG? I think not
#=========================== TEST (try IP Laplace) =========================================================
        mesh = v.function_space().mesh()
        n = fd.FacetNormal(mesh)
        h = fd.CellDiameter(mesh)
#        alpha = 10.
#        S = fd.grad(sols)
#        L = (
#            fd.inner(S, fd.grad(v)) * fd.dx
#            - fd.inner(fd.avg(S), 2*fd.avg(fd.outer(v, n))) * fd.dS
#            - fd.inner(2.*fd.avg(fd.outer(sols,n)), fd.avg(fd.grad(v))) * fd.dS
#            - fd.inner(S, fd.outer(v, n)) * fd.ds
#            + (alpha/fd.avg(h)) * fd.inner(2*fd.avg(fd.outer(sols, n)), 2*fd.avg(fd.outer(v, n))) * fd.dS
#            + (alpha/h) * fd.inner(fd.outer(sols,n), fd.outer(v,n)) * fd.ds
#             )
############# Jan's bilinear form ================================
        h_avg = (h('+') + h('-'))/2
        alpha = 4.0  # hard-coded param
        gamma = 8.0  # hard-coded param
        L = fd.dot(fd.grad(v), fd.grad(sols))*fd.dx \
          - fd.dot(fd.avg(fd.grad(v)), fd.jump(sols, n))*fd.dS \
          - fd.dot(fd.jump(v, n), fd.avg(fd.grad(sols)))*fd.dS \
          + alpha/h_avg*fd.dot(fd.jump(v, n), fd.jump(sols, n))*fd.dS \
          - fd.dot(fd.grad(v), sols*n)*fd.ds \
          - fd.dot(v*n, fd.grad(sols))*fd.ds \
          + gamma/h*v*sols*fd.ds
#=========================== TEST (end) =========================================================
        return L

    def interpolate_initial_guess(self, z):
        x, y = fd.SpatialCoordinate(z.ufl_domain())
        z.interpolate(x**2 * (1-x)**2 * y**2 * (1-y)**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" Just to check things don't break; convergence rate later""")
    parser.add_argument("--disc", choices=["CR","CG","DG"], default="CG")
    parser.add_argument("--smoothing", dest="smoothing", default=False, action="store_true")
    parser.add_argument("--nref", type=int, default=2)
    parser.add_argument("--baseN", type=int, default=16)
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--k", type=int, default=1)
    args, _ = parser.parse_known_args()

    # Initialize values for the constitutive relation
    p = fd.Constant(2.0)
    K = fd.Constant(1.0)

    problem_ = PowerLawTest(args.baseN, p=p, K=K, diagonal=args.diagonal)
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver}[args.disc]
    solver_ = solver_class(problem_, nref=args.nref, smoothing=args.smoothing)

    problem_.interpolate_initial_guess(solver_.z)

    # Choose over which constitutive parameters we do continuation
    K_s = [1.0]
    p_s = [2.0, 2.5]
    p_s = [2.0, 1.9]
    p_s = [2.0]
    continuation_params = {"p": p_s, "K": K_s}

    solver_.solve(continuation_params)

    u = solver_.z
#    u_exact = fd.interpolate(problem_.exact_solution(solver_.Z), fd.FunctionSpace(u.ufl_domain(), "CG", args.k + 3))
#    u_exact.rename("exact_solution")
    u_exact = problem_.exact_solution(solver_.Z)
    print("W^{1, %s} distance to the exact solution = "%float(solver_.p), solver_.W1pnorm(u - u_exact, float(solver_.p)))

    if args.plots:
        fd.File("output/plaw_test.pvd").write(u, u_exact)

