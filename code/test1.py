import firedrake as fd

from main import SmoothingOpVeeserZanotti


def exact_solution(mesh):
    # Since it belongs to CG1, all methods should give the same solution
    W = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    solution = fd.interpolate(
        fd.conditional(fd.ge(x, 0.25),
        fd.conditional(fd.le(x,0.75),
        fd.conditional(fd.ge(y, 0.25),
        fd.conditional(fd.le(y, 0.75), 1., 0.), 0.), 0.), 0.), W)
    solution.rename("exact_solution")
    return solution


def rhs(test_function):
    solution = exact_solution(test_function.ufl_domain())
    rhs = lhs(solution, test_function)
    return rhs


def lhs(u, v):
    if u.function_space().ufl_element().family() == 'Discontinuous Lagrange':
        mesh = u.function_space().mesh()
        n = fd.FacetNormal(mesh)
        h = fd.CellSize(mesh)
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


def solve(mesh, space="CR", smoothing=False):
    assert space in ["CR", "CG", "DG"]
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

    return u


def main():
    mesh = fd.UnitSquareMesh(32, 32, diagonal='crossed')

    u_cg = solve(mesh, "CG")
    u_cr = solve(mesh, "CR")
    u_cr_sth = solve(mesh, "CR", smoothing=True)
    u_dg = solve(mesh, "DG")

    u_dg_interp = fd.project(u_dg, fd.FunctionSpace(mesh, 'CG', 1))
    u_dg_interp.rename(u_dg.name() + '_interp')

    fd.File("test1.pvd").write(exact_solution(u_cg.ufl_domain()), u_cg, u_cr, u_cr_sth, u_dg, u_dg_interp)


if __name__ == '__main__':
    main()
