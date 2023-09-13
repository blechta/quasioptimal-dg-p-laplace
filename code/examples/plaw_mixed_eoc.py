import firedrake as fd
import numpy as np

import argparse
import pprint
import sys
import os
sys.path.append('..')

from solver import NonlinearEllipticProblem_Su, ConformingSolver, CrouzeixRaviartSolver, DGSolver

def compute_rates(errors, res):
    return [np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
                                 for i in range(len(res)-1)]

class PowerLaw(NonlinearEllipticProblem_Su):
    def __init__(self, p, delta, max_shift, alpha=1.01, p_final=0):
        """ Passing p_final only matters when choosing an exact potential, since the exact flux is computed
        through the inverse constitutive relation (and so otherwise the wrong RHS would get computed). If the exact flux
        is chosen instead, this doesn't play a role"""
        super().__init__(p=p,delta=delta,max_shift=max_shift)
        self.alpha = alpha
        self.p_final = p_final

    def mesh(self):
        return fd.Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh")

    def const_rel(self, S):
        return (self.delta**(self.p-1) + fd.inner(S,S)) ** (0.5*(self.p/(self.p-1)) - 1) * S

    def const_rel_inverse(self, D): # This is only really the inverse if delta=0. This is only used when computing the exact flux
        return fd.inner(D, D) ** (0.5*self.p_final - 1) * D

    def exact_potential(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        cutoff = (1 - x*x)*(1 - y*y)
        return cutoff * (x*x + y*y) ** (0.5*self.alpha)

    def exact_flux(self, Z):
        # Compute it from the exact potential
        D = fd.grad(self.exact_potential(Z))
        return self.const_rel_inverse(D)
        # Or define the exact flux directly! We then only have access to the gradient of the potential but that's OK.
        # FIXME: When doing this the wrong solution is computed...
#        x, y = fd.SpatialCoordinate(Z.ufl_domain())
#        return (x*x + y*y) ** (0.5*self.alpha)  * fd.as_vector([1., 1.])

    def rhs(self, z_):
        S_exact = self.exact_flux(z_.function_space())
        _, v = fd.split(z_)
        L = - fd.div(S_exact) * v * fd.dx
        return L

    def interpolate_initial_guess(self, z): # Just choose something non-zero...
        x, y = fd.SpatialCoordinate(z.ufl_domain())
        z.sub(0).interpolate(fd.as_vector([x-2, y+2]))
        z.sub(1).interpolate((x+2)**2 * (2-x)**2 * (y+2)**2 * (2-y)**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" Mixed Formulation""")
    parser.add_argument("--disc", choices=["CR","CG","DG"], default="CG")
    parser.add_argument("--smoothing", dest="smoothing", default=False, action="store_true")
    parser.add_argument("--no-shift", dest="no_shift", default=False, action="store_true")
    parser.add_argument("--nrefs", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=1.01) # Measures how singular is the exact solution; default value should yield linear rate
    parser.add_argument("--penalty", choices=["const_rel","plaw","quadratic","p-d"], default="p-d")
    parser.add_argument("--cr", default="thinning", choices=["newtonian","thinning","thickening"]) # Controls if p is larger or smaller than 2
    parser.add_argument("--p-s", type=int, default=1) # Larger = further away from Newtonian (0 = Newtonian)
    parser.add_argument("--k", type=int, default=1)
    args, _ = parser.parse_known_args()

    # Initialize values for the constitutive relation
    p = fd.Constant(2.0)
    delta = fd.Constant(0.000)
    max_shift = fd.Constant(0.)

    # Choose over which constitutive parameters we do continuation
    # First all the possibilities for p:
    if args.cr == "thinning":
        possible_p_s = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
        args.no_shift = True
    elif args.cr == "thickening":
        possible_p_s = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    else:
        possible_p_s = [2.0]
        args.no_shift = True
    assert (args.p_s <= len(possible_p_s)), "p-s is too large... choose something smaller"
    if args.cr == "newtonian":
        p_s = [2.0]
    else:
        p_s = possible_p_s[:(args.p_s+1)]
    continuation_params = {"p": p_s}

    problem_ = PowerLaw(p=p, delta=delta, max_shift=max_shift, alpha=args.alpha, p_final=p_s[-1])
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver}[args.disc]

    # Choose resolutions
    res = [2**i for i in range(2,args.nrefs+2)]
    h_s = []#[1./re for re in res]

    # To store the errors
    errors = {"F*_flux": [], "F_potential": [], "modular": [], "Lp'_flux": []}#, "Lp_potential": []}

    for nref in range(1, len(res)+1):
        solver_ = solver_class(problem_, nref=nref, smoothing=args.smoothing, no_shift=args.no_shift)

        if (np.abs(float(delta**(p_s[-1]-1))) < 1e-12):
            print(fd.RED % 'Setting initial guess...')
            problem_.interpolate_initial_guess(solver_.z)

        solver_.solve(continuation_params)
        S, u = solver_.z.subfunctions
        Du = fd.grad(u)

        S_exact = problem_.exact_flux(solver_.Z)
        # Use this when choosing the exact potential
        u_exact = problem_.exact_potential(solver_.Z)
        Du_exact = fd.grad(u_exact)
        # Use this when choosing the exact flux
#        Du_exact = problem_.const_rel(S_exact)

        # Compute errors
        quad_degree = 4
        natural_distance_S = solver_.natural_F(w_1=S, w_2=S_exact,
                                               conjugate=True,
                                               quad_degree=quad_degree)
        natural_distance_u = solver_.natural_F(w_1=Du, w_2=Du_exact,
                                               quad_degree=quad_degree)
        errors["F_potential"].append(natural_distance_u)
        errors["F*_flux"].append(natural_distance_S)
        if args.disc == "CG":
            errors["modular"].append(np.nan)
        else:
            modular = solver_.modular(u)
            errors["modular"].append(modular)
#        Lp_error_u = fd.assemble(fd.inner(u-u_exact, u-u_exact)**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
#        errors["Lp_potential"].append(Lp_error_u)
        p_prime_ = p_s[-1]/(p_s[-1]-1.)
        Lp_error_S = fd.assemble(fd.inner(S-S_exact, S-S_exact)**(p_prime_/2.) * fd.dx)**(1/p_prime_)
        errors["Lp'_flux"].append(Lp_error_S)


    convergence_rates = {err_type: compute_rates(errors[err_type], res)
                         for err_type in ["F*_flux", "F_potential", "modular", "Lp'_flux"]}#, "Lp_potential"]}

    print("Mesh sizes (h_max, h_avg): ", h_s)
    print("Computed errors: ")
    pprint.pprint(errors)
    print("Computed rates: ")
    pprint.pprint(convergence_rates)

    # Test
#    fd.File("test.pvd").write(S, fd.Function(fd.VectorFunctionSpace(solver_.z.ufl_domain(), "DG", 1)).interpolate(S_exact))
#    fd.File("test1.pvd").write(fd.project(Du, fd.VectorFunctionSpace(solver_.z.ufl_domain(), "DG", 1)), fd.interpolate(Du_exact, fd.VectorFunctionSpace(solver_.z.ufl_domain(), "DG", 1)))
#    fd.File("test1.pvd").write(u, fd.Function(fd.FunctionSpace(solver_.z.ufl_domain(), "DG", 1)).interpolate(u_exact))
