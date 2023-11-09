import firedrake as fd
import numpy as np

import argparse
import pprint
import os

from solver import NonlinearEllipticProblem, ConformingSolver, CrouzeixRaviartSolver, DGSolver


def compute_rates(errors, res):
    return [np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
                                 for i in range(len(res)-1)]

class PowerLaw(NonlinearEllipticProblem):
    def __init__(self, p, delta, K, max_shift, p_final, beta=1.0):
        super().__init__(p=p, delta=delta, K=K, max_shift=max_shift)
        self.beta = beta
        self.p_final = p_final

    def mesh(self):
        return fd.Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square2.msh")

    def const_rel(self, D):
        return self.K * (self.delta + fd.sqrt(fd.inner(D, D))) ** (self.p-2) * D

    def exact_solution(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        cutoff = (1 - x*x)*(1 - y*y)
        alpha = 1.01 - (2*(1-self.beta))/self.p_final
        return cutoff * (x*x + y*y) ** (0.5*alpha)

    def rhs(self, v):
        sols = self.exact_solution(v.function_space())
        S_ = self.const_rel(fd.grad(sols))
#        L = - fd.div(S_) * v * fd.dx    # This one only makes sense if S_ is regular enough...
        L = fd.inner(S_, fd.grad(v)) * fd.dx   # I suspect in general we need this form... (but note this makes sense only with smoothing)
        return L

    def interpolate_initial_guess(self, z):
        x, y = fd.SpatialCoordinate(z.ufl_domain())
        z.project((x+1)**2 * (1-x)**2 * (y+1)**2 * (1-y)**2)


class PowerLawLDG(PowerLaw):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formulation = "R-u"

    def bcs(self, Z):
        return fd.DirichletBC(Z.sub(1), fd.Constant(0.), "on_boundary")

    def rhs(self, z_):
        sols = self.exact_solution(z_.function_space())
        S_ = self.const_rel(fd.grad(sols))
        _, v = fd.split(z_)
#        L = - fd.div(S_) * v * fd.dx    # This one only makes sense if S_ is regular enough...
        L = fd.inner(S_, fd.grad(v)) * fd.dx   # I suspect in general we need this form... (but note this makes sense only with smoothing)
        return L

    def interpolate_initial_guess(self, z):
        x, y = fd.SpatialCoordinate(z.ufl_domain())
        z.sub(0).interpolate(fd.as_vector([x-2, y+2]))
        z.sub(1).project((x+2)**2 * (2-x)**2 * (y+2)**2 * (2-y)**2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" Primal Formulation""")
    parser.add_argument("--disc", choices=["CR","CG","DG","LDG"], default="CG")
    parser.add_argument("--smoothing", dest="smoothing", default=False, action="store_true")
    parser.add_argument("--no-shift", dest="no_shift", default=False, action="store_true")
    parser.add_argument("--nrefs", type=int, default=6)
    parser.add_argument("--beta", type=float, default=1.0) # Expected rate of convergence
    parser.add_argument("--penalty", choices=["const_rel","plaw","quadratic","p-d"], default="p-d")
    parser.add_argument("--cr", default="thinning", choices=["newtonian","thinning","thickening"]) # Controls if p is larger or smaller than 2
    parser.add_argument("--p-s", type=int, default=1) # Larger = further away from Newtonian (0 = Newtonian)
    parser.add_argument("--k", type=int, default=1)
    args, _ = parser.parse_known_args()

    # Initialize values for the constitutive relation
    p = fd.Constant(2.0)
    K = fd.Constant(1.0)
    delta = fd.Constant(0.001)
    max_shift = fd.Constant(0.)

    # Choose over which constitutive parameters we do continuation
    # First all the possibilities for p:
    if args.cr == "thinning":
        possible_p_s = [2.0, 1.9, 1.7, 1.6, 1.5]
        args.no_shift = True
    elif args.cr == "thickening":
        possible_p_s = [2.0, 2.5, 3.0, 4.0, 4.5]
    else:
        possible_p_s = [2.0]
        args.no_shift = True
    assert (args.p_s <= len(possible_p_s)), "p-s is too large... choose something smaller"
    if args.cr == "newtonian":
        p_s = [2.0]
    else:
        p_s = possible_p_s[:(args.p_s+1)]
    continuation_params = {"p": p_s}


    problem_class = {"DG": PowerLaw,
                     "CG": PowerLaw,
                     "CR": PowerLaw,
                     "LDG": PowerLawLDG}[args.disc]
    problem_ = problem_class(p=p, delta=delta, K=K, max_shift=max_shift, p_final=p_s[-1], beta=args.beta)
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver,
                    "LDG": DGSolver}[args.disc]
    solver_args = {"nref": args.nrefs, "smoothing": args.smoothing}
    if args.disc in ["CR","DG", "LDG"]: solver_args["penalty_form"] = args.penalty
    if args.disc == "CG": args.no_shift = True


    # Choose resolutions
    res = [2**i for i in range(1, args.nrefs+2)]
    h_s = []

    # To store the errors
    errors = {"F": [], "modular": [], "Lp": [], "total": []}

    for nref in range(len(res)):
        solver_kwargs = {"nref": nref, "smoothing": args.smoothing, "no_shift": args.no_shift}
        solver_ = solver_class(problem_, **solver_kwargs)

        problem_.interpolate_initial_guess(solver_.z)

        solver_.solve(continuation_params)
        if args.disc == "LDG":
            R, u = solver_.z.subfunctions
        else:
            u = solver_.z

        u_exact = problem_.exact_solution(solver_.Z)

        # Compute current mesh size
        h = fd.Function(fd.FunctionSpace(solver_.z.ufl_domain(), "DG", 0)).interpolate(fd.CellSize(solver_.z.ufl_domain()))
        with h.dat.vec_ro as w:
            h_s.append((w.max()[1], w.sum()/w.getSize()))

        # Compute errors
        if args.disc == "LDG":
            natural_distance = solver_.natural_F(w_1=fd.grad(u)+R, w_2=fd.grad(u_exact))
        else:
            natural_distance = solver_.natural_F(w_1=fd.grad(u), w_2=fd.grad(u_exact))
        errors["F"].append(natural_distance)
        if args.disc == "CG":
            errors["modular"].append(np.nan)
            total_error = 0.0
        else:
            modular = solver_.modular(u)
            errors["modular"].append(modular)
            total_error = modular
        Lp_error = fd.assemble(fd.inner(u-u_exact, u-u_exact)**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
        errors["Lp"].append(Lp_error)
        total_error += natural_distance
        errors["total"].append(total_error)


    convergence_rates = {err_type: compute_rates(errors[err_type], res)
                         for err_type in ["F", "modular", "Lp", "total"]}

    print("Mesh sizes (h_max, h_avg): ", h_s)
    print("Computed errors: ")
    pprint.pprint(errors)
    print("Computed rates: ")
    pprint.pprint(convergence_rates)
    print("Average EOC:   ", sum(convergence_rates["total"])/len(convergence_rates["total"]))
