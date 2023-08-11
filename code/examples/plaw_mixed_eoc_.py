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

#def natural_F(w_1, w_2, p_, delta_, conjugate=False):
#    if conjugate: # Computes the natural distance with p', delta^{p-1}
#        delta_ = delta_**(p_ - 1.)
#        p_ = p_/(p_-1.)
#
#    F_ = (delta_ + fd.inner(w_1, w_1)**(1/2.))**(0.5*float(p_)- 1) * w_1
#    F_ -= (delta_ + fd.inner(w_2, w_2)**(1/2.))**(0.5*float(p_)- 1) * w_2 # I don't know why this line breaks things...
#    return (fd.assemble(fd.inner(F_, F_) * fd.dx))**0.5
#

class PowerLaw(NonlinearEllipticProblem_Su):
    def __init__(self, baseN, p, delta, K, diagonal=None):
        super().__init__(p=p, delta=delta, K=K)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self):
        return fd.UnitSquareMesh(self.baseN, self.baseN, diagonal=self.diagonal)

    def const_rel(self, S):
        p_prime = self.p/(self.p-1.)
        delta_prime = self.delta**(self.p -1)
        return self.K * (delta_prime + fd.inner(S, S)) ** (0.5*float(p_prime) - 1) * S # Without the "float" here sometimes things crash even for p=2...

    def const_rel_inverse(self, D):
        return self.K * (self.delta + fd.inner(D, D)) ** (0.5*self.p - 1) * D

    def exact_potential(self, Z):
        x, y = fd.SpatialCoordinate(Z.ufl_domain())
        return fd.sin(4*fd.pi*x) * y**2 * (1.-y)**2

    def exact_flux(self, Z): # TODO: This only works if delta = 0, because we know the inverse constitutive relation... what to do in general?
        D = fd.grad(self.exact_potential(Z))
        return self.const_rel_inverse(D)

    def rhs(self, z_):
        S_exact = self.exact_flux(z_.function_space())
        _, v = fd.split(z_)
        L = - fd.div(S_exact) * v * fd.dx
#        L = fd.inner(S, fd.grad(v)) * fd.dx # Should this work for DG? I think not
        return L

    def interpolate_initial_guess(self, z):
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
    K = fd.Constant(1.0)
    delta = fd.Constant(0.0)

    problem_ = PowerLaw(args.baseN, p=p, delta=delta, K=K, diagonal=args.diagonal)
    solver_class = {"CG": ConformingSolver,
                    "CR": CrouzeixRaviartSolver,
                    "DG": DGSolver}[args.disc]

    # Choose over which constitutive parameters we do continuation
    delta_s = [0.0]
    K_s = [1.0]
    p_s = [2.0, 2.5, 3.0]
    p_s = [2.0, 1.8]#, 1.7]
    p_s = [2.0]
    continuation_params = {"p": p_s, "K": K_s}

    # Choose resolutions TODO: Unstructured mesh?
    res = [2**i for i in range(2,args.nrefs+2)]
    h_s = [1./re for re in res]

    # To store the errors
    errors = {"F*_flux": [], "F_potential": [], "modular": [], "Lp'_flux": [], "Lp_potential": []}

    for nref in range(1, len(res)+1):
        solver_ = solver_class(problem_, nref=nref, smoothing=args.smoothing)

        problem_.interpolate_initial_guess(solver_.z)

        solver_.solve(continuation_params)
#        S, u = solver_.z.subfunctions # Which is the correct one?
        S, u = fd.split(solver_.z)

        u_exact = problem_.exact_potential(solver_.Z)
        S_exact = problem_.exact_flux(solver_.Z)

        # Compute errors
        # Compute explicitly the p=2 case for testing
        natural_distance_S = fd.assemble(fd.inner(S-S_exact, S-S_exact)**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
        natural_distance_u = fd.assemble(fd.inner(fd.grad(u-u_exact), fd.grad(u-u_exact))**(p_s[-1]/2.) * fd.dx)**(1/p_s[-1])
        # But these are the ones we want and don't work for some reason...
#        natural_distance_u = solver_.natural_F(w_1=fd.grad(u), w_2=fd.grad(u_exact)) # This one is correct for p=2 (but e.g. produces "nan" for CG)
#        natural_distance_S = solver_.natural_F(w_1=S, w_2=S_exact, conjugate=True)  # This one sometimes just crashes... seems incorrect for p != 2
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
