import firedrake as fd
import numpy as np

import argparse
import pprint
import sys
sys.path.append('..')

from solver import NonlinearEllipticProblem, ConformingSolver, CrouzeixRaviartSolver, DGSolver

def compute_rates(errors, res):
    return [np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
                                 for i in range(len(res)-1)]

class PowerLaw(NonlinearEllipticProblem):
    def __init__(self, baseN, p, delta, K, diagonal=None):
        super().__init__(p=p, delta=delta, K=K)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self):
        return fd.UnitSquareMesh(self.baseN, self.baseN, diagonal=self.diagonal)

    def const_rel(self, D):
        return self.K * (self.delta + fd.inner(D, D)) ** (0.5*self.p-1) * D

    def exact_solution(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        return fd.sin(4*fd.pi*x) * y**2 * (1.-y)**2

    def rhs(self, v):
        sols = self.exact_solution(v.function_space())
        S = self.const_rel(fd.grad(sols))
        L = - fd.div(S) * v * fd.dx
#        L = fd.inner(S, fd.grad(v)) * fd.dx # Should this work for DG? I think not
        return L

    def interpolate_initial_guess(self, z):
        x, y = fd.SpatialCoordinate(z.ufl_domain())
        z.interpolate(x**2 * (1-x)**2 * y**2 * (1-y)**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" Just to check things don't break; convergence rate later""")
    parser.add_argument("--disc", choices=["CR","CG","DG"], default="CG")
    parser.add_argument("--smoothing", dest="smoothing", default=False, action="store_true")
    parser.add_argument("--nrefs", type=int, default=6)
    parser.add_argument("--baseN", type=int, default=16)
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--k", type=int, default=1)
    args, _ = parser.parse_known_args()

    # Initialize values for the constitutive relation
    p = fd.Constant(2.0)
    K = fd.Constant(1.0)
    delta = fd.Constant(0.001)

    problem_ = PowerLaw(args.baseN, p=p, delta=delta, K=K, diagonal=args.diagonal)
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver}[args.disc]

    # Choose over which constitutive parameters we do continuation
    delta_s = [0.001]
    K_s = [1.0]
    p_s = [2.0, 2.5, 3.0]
    p_s = [2.0, 1.8, 1.7]
#    p_s = [2.0]
    continuation_params = {"p": p_s, "K": K_s}

    # Choose resolutions TODO: Unstructured mesh?
    res = [2**i for i in range(2,args.nrefs+2)]
    h_s = [1./re for re in res]

    # To store the errors
    errors = {"F": [], "modular": [], "Lp": []}

    for nref in range(1, len(res)+1):
        solver_ = solver_class(problem_, nref=nref, smoothing=args.smoothing)

        problem_.interpolate_initial_guess(solver_.z)

        solver_.solve(continuation_params)
        u = solver_.z

        u_exact = problem_.exact_solution(solver_.Z)

        # Compute errors
        natural_distance = solver_.natural_F(w_1=fd.grad(u), w_2=fd.grad(u_exact))
        errors["F"].append(natural_distance)
        if args.disc == "CG":
            errors["modular"].append(np.nan)
        else:
            modular = solver_.modular(u)
            errors["modular"].append(modular)
        Lp_error = fd.assemble(fd.inner(u-u_exact, u-u_exact)**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
        errors["Lp"].append(Lp_error)


    convergence_rates = {err_type: compute_rates(errors[err_type], res)
                         for err_type in ["F", "modular", "Lp"]}

    print("Computed errors: ")
    pprint.pprint(errors)
    print("Computed rates: ")
    pprint.pprint(convergence_rates)
