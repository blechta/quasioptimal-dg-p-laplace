import firedrake as fd
import numpy as np

import argparse
import pprint

from smoothing import SmoothingOpVeeserZanotti

def cutoff(r):
    smoothing_parameter = 100.
    Heaviside1 = 0.5 + 0.5 * smoothing_parameter * (r-0.25) / fd.sqrt(1 + (smoothing_parameter*(r-0.25))**2)
    Heaviside2 = 0.5 + 0.5 * smoothing_parameter * (0.75-r) / fd.sqrt(1 + (smoothing_parameter*(0.75-r))**2)
    return Heaviside1*Heaviside2

def exact_solution(mesh, smooth=False): #FIXME: What's a good way of deciding which solution? rhs doesn't take additional parameters
    # Smooth exact solutions
    x, y = fd.SpatialCoordinate(mesh)
    if smooth:
        solution = fd.sin(4*fd.pi*x)*y**2*(1.-y)**2
    else:
        #TODO: This should belong to H1 and no better. Easier to test things in between first?
        #FIXME: Does not work yet... the singularity is probably causing trouble
        r = ((x-0.5)**2 + (y-0.5)**2)**0.5
        solution = fd.ln(fd.ln(r))*cutoff(r)
        #============== TEST =========================
        fd.File("test_eoc.pvd").write(fd.interpolate(solution, fd.FunctionSpace(mesh, "CG",1)))
        #=============================================
    return solution

def lhs(u, v):
    if v.function_space().ufl_element().family() == 'Discontinuous Lagrange':
        mesh = u.function_space().mesh()
        n = fd.FacetNormal(mesh)
        h = fd.CellDiameter(mesh)
        h_avg = (h('+') + h('-'))/2
        alpha = 4.0  # hard-coded param
        gamma = 8.0  # hard-coded param
        a = fd.dot(fd.grad(v), fd.grad(u))*fd.dx \
          - fd.dot(fd.avg(fd.grad(v)), fd.jump(u, n))*fd.dS \
          - fd.dot(fd.jump(v, n), fd.avg(fd.grad(u)))*fd.dS \
          + alpha/h_avg*fd.dot(fd.jump(v, n), fd.jump(u, n))*fd.dS \
          - fd.dot(fd.grad(v), u*n)*fd.ds \
          - fd.dot(v*n, fd.grad(u))*fd.ds \
          + gamma/h*v*u*fd.ds
    else:
        a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
    return a

def rhs(test_function):
    solution = exact_solution(test_function.ufl_domain())
    rhs = lhs(solution, test_function)
    return rhs

def compute_traces(u):
    interior = fd.jump(u)**2 * fd.dS
    exterior = u**2 * fd.ds
    return fd.assemble(interior)**0.5, fd.assemble(exterior)**0.5

def compute_errors(u):
    u_ex = exact_solution(u.ufl_domain())
    err_l2 = (u-u_ex)**2 * fd.dx
    err_h1 = fd.dot(fd.grad(u-u_ex), fd.grad(u-u_ex)) * fd.dx
    return fd.assemble(err_l2)**0.5, fd.assemble(err_h1)**0.5

def compute_rates(errors, res):
    return [np.log(errors[i]/errors[i+1])/np.log(res[i+1]/res[i])
                                 for i in range(len(res)-1)]

def solve_laplace(resolution, space="CR", smoothing=False):
    assert space in ["CR", "CG", "DG"]
    mesh = fd.UnitSquareMesh(resolution, resolution)
    Z = fd.FunctionSpace(mesh, space, 1)

    function_name = "u_" + space + ("_sth" if smoothing else "")
    u = fd.Function(Z, name=function_name)
    v = fd.TestFunction(Z)

    if u.function_space().ufl_element().family() == 'Discontinuous Lagrange':
        bcs = None
    else:
        bcs = fd.DirichletBC(Z, fd.Constant(0.), "on_boundary")

    if smoothing:
        op = SmoothingOpVeeserZanotti(Z)
        b = op.apply(rhs)
    else:
        b = fd.assemble(rhs(v))

    a = lhs(u, v)
    A = fd.assemble(fd.derivative(a, u), bcs=bcs)
    fd.solve(A, u, b)

    j, t = compute_traces(u)
    err_l2, err_h1 = compute_errors(u)

    errors = [j,t,err_l2,err_h1]

    return u, errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Test the order of convergence for Laplace with a smooth exact solution.""")
    parser.add_argument("--disc", choices=["CR","CG","DG"], default="CR")
    parser.add_argument("--smoothing", dest="smoothing", default=False, action="store_true")
    parser.add_argument("--nrefs", type=int, default=6)
    args = parser.parse_args()

    res = [2**i for i in range(2,args.nrefs)]
    h_s = [1./re for re in res]


    # Compute the errors and rates
    errors = {n: solve_laplace(res[n], space=args.disc, smoothing=args.smoothing)[1] for n in range(len(res))}

    # Rearrange
    errors2 = {"jumps": [errors[j][0] for j in range(len(res))],
               "trace": [errors[j][1] for j in range(len(res))],
               "L2": [errors[j][2] for j in range(len(res))],
               "H1": [errors[j][3] for j in range(len(res))]}

    convergence_rates = {err_type: compute_rates(errors2[err_type], res)
                         for err_type in ["jumps", "trace", "L2", "H1"]}
    print("Computed errors: ")
    pprint.pprint(errors2)
    print("Computed rates: ")
    pprint.pprint(convergence_rates)
