# Supporting software for article _Quasi-optimal Discontinuous Galerkin discretisations of the $p$-Dirichlet problem_ #

This is a supporting code for article _Quasi-optimal Discontinuous Galerkin
discretisations of the $p$-Dirichlet problem_ by J. Blechta, P. A.
Gazca-Orozco, A. Kaltenbach, and M. Růžička.

It contains [an implementation](src/smoothing.py) of the smoothing operator by
A. Veeser and P. Zanotti from:

* [doi:10.1137/17M1151651](https://doi.org/10.1137/17M1151651),
* [doi:10.1137/17M1151675](https://doi.org/10.1137/17M1151675)

using [Firedrake](https://www.firedrakeproject.org) and
[PETSc](https://petsc.org). Only the lowest-order case on triangles has been
so far implemented.


## Dependencies ##

The code can be run with Firedrake components given by
[doi:10.5281/zenodo.10061067](https://doi.org/10.5281/zenodo.10061067) plus
[Gmsh 4.8.4](https://gmsh.info). The very same version of the software stack is
available as
[`firedrakeproject/firedrake:2023-10`](https://hub.docker.com/layers/firedrakeproject/firedrake/2023-10/images/sha256-65ca53d448cb4ac79beebede816092acf36ff54087f514bbdbec46afe978c98a)
image from [Docker Hub](https://hub.docker.com/r/firedrakeproject/firedrake/tags).
A convenience script [run_firedrake_container](bin/run_firedrake_container) is
available to spawn a Docker container. This is a thin wrapper on top of
[`docker run` command](https://docs.docker.com/engine/reference/run/).


## Reproducing the paper results ##

The compute-heavy part of the experiments is run using the
Firedrake stack described above by running:

```shell
cd src/
make -j <n>
```

Finest mesh refinements need around 40 GB of RAM. One can
edit [Makefile](src/Makefile) and decrease `--nrefs 6` by one
or two levels. This will also provide significant speed-up
as a sparse direct solver is used.

This produces log files in `src/output/`. Tables and figures
for the paper are produced by running:
```shell
python src/postprocess.py src/output/
python src/plot_mesh.py
```


## Stand-alone usage of the smoothing operator ##

The smoothing operators (a variant for the Crouzeix–Raviart element
and a variant for the DG degree 1 element) are available
through class `SmoothingOpVeeserZanotti` implemented in
[smoothing.py](src/smoothing.py). The operator can directly be used
to assemble right-hand sides with smoothed test functions, as indicated
in [the `__main__` section of smoothing.py](src/smoothing.py#L314):

```python
mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, 'CR', 1)
op = SmoothingOpVeeserZanotti(V)

def rhs(test_function):
    f = as_vector([1, -1])
    return inner(f, grad(test_function))*dx
b = op.apply(rhs)
```

Alternatively, the smoothing operator can be hooked to right-hand side assembly
in `firedrake.NonlinearVariationalSolver`. This is shown in
[`NonlinearEllipticSolver.nonlinear_variational_solver()`](src/solver.py#L125).


## Documentation ##

The derivation of some formulas used to implement
[smoothing.py](src/smoothing.py) can be found in [a brief technical
report](doc/main.pdf). The construction of the smoothing operator
and full-fledged numerical analysis for linear problems:

* [doi:10.1137/17M1151651](https://doi.org/10.1137/17M1151651),
* [doi:10.1137/17M1151675](https://doi.org/10.1137/17M1151675)

Generalization for non-linear problems:

*  J. Blechta, P. A. Gazca-Orozco, A. Kaltenbach, and M. Růžička.
   _Quasi-optimal Discontinuous Galerkin discretisations of the $p$-Dirichlet
   problem_. To be submitted. 2023.


## License ##

Copyright 2023 Jan Blechta, Alexei Gazca

[The MIT License](LICENSE)
