import firedrake as fd

from main import SmoothingOpVeeserZanotti

# Computes the viscosity/conductivity for p-Laplace. For now I'll just use p=2...
def viscosity(z, p_):
    return fd.inner(fd.grad(z), fd.grad(z)) ** ((p_-2.)/2.)

# Defines the exact solution. Since it belongs to CG1, all methods should
# give the same solution
# (I just wanted it to be one for max(abs(x-0.5), abs(y-0.5)) < 0.25 and zero otherwise
# Is there a better way to define this?)
def exact_solution(mesh):
    W = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    solution = fd.interpolate(fd.conditional(fd.ge(x, 0.25),
                             fd.conditional(fd.le(x,0.75),
                                            fd.conditional(fd.ge(y, 0.25),
                                                           fd.conditional(fd.le(y, 0.75), 1., 0.), 0.), 0.), 0.), W)
    solution.rename("exact_solution")

    return solution

# Defines the appropriate right-hand-side.
def rhs(test_function):
    solution = exact_solution(test_function.ufl_domain())
    visc = viscosity(solution, 2.0) #FIXME: Set this from outside...
    rhs = fd.inner(visc * fd.grad(solution), fd.grad(test_function)) * fd.dx
    return rhs

# Solve the problem, possibly with smoothing operator on the rhs
def solve(mesh, p_, space="CR", smoothing=False):
    assert space in ["CR", "CG"], "This only works with CG or CR elements..."
    Z = fd.FunctionSpace(mesh, space, 1)

    function_name = "u_"+space
    if smoothing: function_name += "_sth"
    u = fd.Function(Z, name=function_name)
    v = fd.TestFunction(Z)

    # Zero BCs. What happens when BCs are not zero?
    bcs = fd.DirichletBC(Z, fd.Constant(0.), "on_boundary")

    # Get right-hand-side and apply smoothing if required
    if smoothing:
        op = SmoothingOpVeeserZanotti(Z)
        b = op.apply(rhs)
    else:
        #F_rhs = rhs(v)
        b = fd.assemble(rhs(v))

    ## Define form
    #visc = viscosity(u, p_)
    #F = fd.inner(visc * fd.grad(u), fd.grad(v)) * fd.dx
    # FIXME: Probably some dumb mistake, but the solver diverges when visc is there (even for p=2)
    #F = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
    #F -= F_rhs
    #
    ## Solver parameters
    #sp = {"snes_type": "newtonls",
    #      "snes_monitor": None,
    #      "snes_converged_reason": None,
    #      "snes_linesearch_type": "nleqerr",
    #      'snes_linesearch_damping': 1.0,
    #      "snes_atol": 1.0e-7,
    #      "snes_rtol": 1.0e-7,
    #      "ksp_type": "preonly",
    #      "pc_type": "lu",
    #      "pc_factor_mat_solver_type": "mumps",
    #      "mat_mumps_cntl_1": 1e-6,
    #      "mat_mumps_icntl_14": 2000}
    #
    #fd.solve(F == 0, u, bcs, solver_parameters=sp)

    #visc = viscosity(u, p_)
    #a = fd.inner(visc * fd.grad(u), fd.grad(v)) * fd.dx
    # FIXME: Probably some dumb mistake, but the solver diverges when visc is there (even for p=2)
    a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
    A = fd.assemble(fd.derivative(a, u), bcs=bcs)
    fd.solve(A, u, b)

    return u

if __name__ == '__main__':
    # Choose the exponent for p-Laplace
    p_ = fd.Constant(2.)
    mesh = fd.UnitSquareMesh(32, 32, diagonal='crossed')

    # Compare the solutions with CG1, CR1 and CR1 with smoothing
    u_cg = solve(mesh, p_, "CG")
    u_cr = solve(mesh, p_, "CR")
    u_cr_sth = solve(mesh, p_, "CR", smoothing=True)

    # Visualize. All these functions should coincide
    fd.File("test1.pvd").write(exact_solution(u_cg.ufl_domain()), u_cg, u_cr, u_cr_sth)
