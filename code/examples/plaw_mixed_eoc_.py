import firedrake as fd
import numpy as np

import argparse
import pprint
import sys
sys.path.append('..')

from solver import NonlinearEllipticProblem_Su, ConformingSolver, CrouzeixRaviartSolver, DGSolver

def compute_rates(errors, res):
    return [np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
                                 for i in range(len(res)-1)]

def compute_error(z, S_ex, u_ex, p_): # TODO: Why do we get in trouble without that regularisation?
    S, u = z.subfunctions
    F_u = (1e-12 + fd.inner(fd.grad(u), fd.grad(u)))**(0.25*p_ - 0.5) * fd.grad(u)
    F_u -= (1e-12 + fd.inner(fd.grad(u_ex), fd.grad(u_ex)))**(0.25*p_ - 0.5) * fd.grad(u_ex)
    F_S = (1e-12 + fd.inner(S, S))**(0.25*(p_/(p_-1)) - 0.5) * S
    F_S -= (1e-12 + fd.inner(S_ex, S_ex))**(0.25*(p_/(p_-1)) - 0.5) * S_ex
    natural_d_u = fd.assemble(fd.inner(F_u, F_u) * fd.dx)**0.5
    natural_d_S = fd.assemble(fd.inner(F_S, F_S) * fd.dx)**0.5
    return natural_d_S, natural_d_u

class PowerLaw(NonlinearEllipticProblem_Su):
    def __init__(self, baseN, p, p_final, diagonal=None):
        super().__init__(p=p)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN
        self.p_final = p_final

    def mesh(self):
        return fd.UnitSquareMesh(self.baseN, self.baseN, diagonal=self.diagonal)

    def const_rel(self, S):
        return fd.inner(S,S) ** (0.5*(self.p/(self.p-1)) - 1) * S

    def const_rel_inverse(self, D): # We just use this for the exact solution so we need to use the final value of "p"
        return fd.inner(D, D) ** (0.5*self.p_final - 1) * D

    def exact_potential(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        return fd.sin(4*fd.pi*x) * y**2 * (1.-y)**2

    def exact_flux(self, Z):
        D = fd.grad(self.exact_potential(Z))
        return self.const_rel_inverse(D)

    def rhs(self, z_):
        S_exact = self.exact_flux(z_.function_space())
        _, v = fd.split(z_)
        L = - fd.div(S_exact) * v * fd.dx
#        L = fd.inner(S, fd.grad(v)) * fd.dx # Should this work for DG? I think not
        return L

    def interpolate_initial_guess(self, z): # Just choose something non-zero...
        x, y = fd.SpatialCoordinate(z.ufl_domain())
        z.sub(0).interpolate(fd.as_vector([x-1, y+1]))
        z.sub(1).interpolate(x**2 * (1-x)**2 * y**2 * (1-y)**2)

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

    # Choose over which constitutive parameters we do continuation
    p_s = [2.0, 2.5, 3.0]
#    p_s = [2.0, 1.8]#, 1.7]
#    p_s = [2.0]
    continuation_params = {"p": p_s}

    problem_ = PowerLaw(args.baseN, p=p, p_final=p_s[-1], diagonal=args.diagonal)
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver}[args.disc]


    # Choose resolutions TODO: Unstructured mesh?
    res = [2**i for i in range(2,args.nrefs+2)]
    h_s = [1./re for re in res]

    # To store the errors
    errors = {"F*_flux": [], "F_potential": [], "modular": [], "Lp'_flux": [], "Lp_potential": []}

    for nref in range(1, len(res)+1):
        solver_ = solver_class(problem_, nref=nref, smoothing=args.smoothing)

        problem_.interpolate_initial_guess(solver_.z)

        solver_.solve(continuation_params)
        S, u = solver_.z.subfunctions # Which is the correct one?
#        S, u = fd.split(solver_.z)

        u_exact = problem_.exact_potential(solver_.Z)
        S_exact = problem_.exact_flux(solver_.Z)

        a = fd.assemble(fd.dot(S-S_exact, S-S_exact)*fd.dx)

        # Compute errors
        # Compute explicitly the p=2 case for testing
#        natural_distance_S = fd.assemble(fd.inner(S-S_exact, S-S_exact)**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
#        natural_distance_u = fd.assemble(fd.inner(fd.grad(u-u_exact), fd.grad(u-u_exact))**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
        # Let's try 
        natural_distance_S, natural_distance_u = compute_error(solver_.z, S_exact, u_exact, p_s[-1])
        errors["F_potential"].append(natural_distance_u)
        errors["F*_flux"].append(natural_distance_S)
        if args.disc == "CG":
            errors["modular"].append(np.nan)
        else:
            modular = solver_.modular(u)
            errors["modular"].append(modular)
        Lp_error_u = fd.assemble(fd.inner(u-u_exact, u-u_exact)**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
        errors["Lp_potential"].append(Lp_error_u)
        p_prime_ = p_s[-1]/(p_s[-1]-1.)
        Lp_error_S = fd.assemble(fd.inner(S-S_exact, S-S_exact)**(p_prime_/2.) * fd.dx)**(1/p_prime_)
        errors["Lp'_flux"].append(Lp_error_S)


    convergence_rates = {err_type: compute_rates(errors[err_type], res)
                         for err_type in ["F*_flux", "F_potential", "modular", "Lp'_flux", "Lp_potential"]}

    fd.warning(fd.BLUE % "We are computing the wrong thing for F*_flux")
    print("Computed errors: ")
    pprint.pprint(errors)
    print("Computed rates: ")
    pprint.pprint(convergence_rates)

    # Testing...
#    u.rename("Velocity")
#    fd.File("test.pvd").write(u, fd.Function(solver_.Z.sub(1)).interpolate(u_exact))
