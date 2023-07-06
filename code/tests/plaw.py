import firedrake as fd
#import numpy as np

import argparse
import sys
sys.path.append('..')

# importing
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
        return self.K * fd.inner(D, D) ** ((self.p - 2)/2.) * D

    def exact_solution(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        return fd.sin(4*fd.pi*x) * y**2 * (1.-y)**2

    def rhs(self, v):
        sols = self.exact_solution(v.function_space())
        S = self.const_rel(fd.grad(sols))
        L = - fd.div(S) * v * fd.dx
#        L = fd.inner(S, fd.grad(v)) * fd.dx  # This one doesn't work for DG (which makes sense...)
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

    p = 2.0
    K = 1.0

    problem_ = PowerLawTest(args.baseN, p, K, diagonal=args.diagonal)
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver}[args.disc]
    solver_ = solver_class(problem_, nref=args.nref, smoothing=args.smoothing)

    problem_.interpolate_initial_guess(solver_.z)

    solver_.solve()

    u = solver_.z
    u_exact = fd.interpolate(problem_.exact_solution(solver_.Z), fd.FunctionSpace(u.ufl_domain(), "CG", args.k + 3))
    u_exact.rename("exact_solution")
    print("W^{1, %s} distance to the exact solution = "%p, solver_.W1pnorm(u - u_exact, p))

    if args.plots:
        fd.File("output/plaw_test.pvd").write(u, u_exact)
