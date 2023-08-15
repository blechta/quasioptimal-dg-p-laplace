import firedrake as fd
import numpy as np

import argparse
import pprint
import sys
import os
sys.path.append('..')

from solver import NonlinearEllipticProblem, ConformingSolver, CrouzeixRaviartSolver, DGSolver

def compute_rates(errors, res):
    return [np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
                                 for i in range(len(res)-1)]

class PowerLaw(NonlinearEllipticProblem):
    def __init__(self, p, delta, K, alpha=1.01):
        super().__init__(p=p, delta=delta, K=K)
        self.alpha = alpha

    def mesh(self):
        return fd.Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh")

    def const_rel(self, D):
        return self.K * (self.delta + fd.inner(D, D)) ** (0.5*self.p-1) * D

    def exact_solution(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        cutoff = (1 - x*x)*(1 - y*y)
        return cutoff * (x*x + y*y) ** (0.5*self.alpha)

    def rhs(self, v):
        sols = self.exact_solution(v.function_space())
        S = self.const_rel(fd.grad(sols))
        L = - fd.div(S) * v * fd.dx
#        L = fd.inner(S, fd.grad(v)) * fd.dx # Should this work for DG? I think not
        return L

    def interpolate_initial_guess(self, z):
        x, y = fd.SpatialCoordinate(z.ufl_domain())
        z.interpolate((x+1)**2 * (1-x)**2 * (y+1)**2 * (1-y)**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" Just to check things don't break; convergence rate later""")
    parser.add_argument("--disc", choices=["CR","CG","DG"], default="CG")
    parser.add_argument("--smoothing", dest="smoothing", default=False, action="store_true")
    parser.add_argument("--nrefs", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1.01) # Measures how singular is the exact solution; default value should yield linear rate
    parser.add_argument("--penalty", choices=["const_rel","plaw","quadratic"], default="const_rel")
    parser.add_argument("--cr", default="thinning", choices=["newtonian","thinning","thickening"]) # Controls if p is larger or smaller than 2
    parser.add_argument("--p-s", type=int, default=1) # Larger = further away from Newtonian (0 = Newtonian)
    parser.add_argument("--k", type=int, default=1)
    args, _ = parser.parse_known_args()

    # Initialize values for the constitutive relation
    p = fd.Constant(2.0)
    K = fd.Constant(1.0)
    delta = fd.Constant(0.0001)

    problem_ = PowerLaw(p=p, delta=delta, K=K, alpha=args.alpha)
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver}[args.disc]
    solver_args = {"nref": args.nrefs, "smoothing": args.smoothing}
    if args.disc in ["CR","DG"]: solver_args["penalty_form"] = args.penalty

    # Choose over which constitutive parameters we do continuation
    # First all the possibilities for p:
    if args.cr == "thinning":
        possible_p_s = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
    elif args.cr == "thickening":
        possible_p_s = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    else:
        possible_p_s = [2.0]
    assert (args.p_s <= len(possible_p_s)), "p-s is too large... choose something smaller"
    delta_s = [0.001]
    K_s = [1.0]
    if args.cr == "newtonian":
        p_s = [2.0]
    else:
        p_s = possible_p_s[:(args.p_s+1)]
    continuation_params = {"p": p_s, "K": K_s}

    # Choose resolutions
    res = [2**i for i in range(2,args.nrefs+2)]
    h_s = []#[1./re for re in res]

    # To store the errors
    errors = {"F": [], "modular": [], "Lp": []}

    for nref in range(1, len(res)+1):
        solver_args["nref"] = nref
        solver_ = solver_class(problem_, **solver_args)

        if (np.abs(float(delta)) < 1e-10): problem_.interpolate_initial_guess(solver_.z)

        solver_.solve(continuation_params)
        u = solver_.z

        u_exact = problem_.exact_solution(solver_.Z)

        # Compute current mesh size
        h = fd.Function(fd.FunctionSpace(solver_.z.ufl_domain(), "DG", 0)).interpolate(fd.CellSize(solver_.z.ufl_domain()))
        with h.dat.vec_ro as w:
            h_s.append((w.max()[1], w.sum()/w.getSize()))

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

    print("Mesh sizes (h_max, h_avg): ", h_s)
    print("Computed errors: ")
    pprint.pprint(errors)
    print("Computed rates: ")
    pprint.pprint(convergence_rates)
