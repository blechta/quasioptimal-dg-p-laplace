from firedrake import *
from petsc4py import PETSc
import numpy as np


class SmoothingOpBase(object):

    def __init__(self, mesh):
        assert mesh.topological_dimension() == 2, '3d not yet implemented'
        self.dm = mesh.topology_dm
        self.v_start, self.v_end = mesh.topology_dm.getDepthStratum(0)
        self.c_start, self.c_end = mesh.topology_dm.getHeightStratum(0)
        self.f_start, self.f_end = mesh.topology_dm.getHeightStratum(1)

    def vertices(self, cell):
        assert cell >= self.c_start and cell < self.c_end
        return [v for v in self.dm.getAdjacency(cell)
                if v >= self.v_start and v < self.v_end]

    def cells(self, vertex):
        assert vertex >= self.v_start and vertex < self.v_end
        return [c for c in self.dm.getAdjacency(vertex)
                if c >= self.c_start and c < self.c_end]

    def facets(self, entity):
        return [f for f in self.dm.getAdjacency(entity)
                if f >= self.f_start and f < self.f_end]

    def cone(self, entity):
        return self.dm.getCone(entity)

    def first_cell(self, vertex):
        return self.cells(vertex)[0]

    def test_vertex_cell_conn(self):
        for c in range(self.c_start, self.c_end):
            assert c in (c1 for v in self.vertices(c) for c1 in self.cells(v))
        for v in range(self.v_start, self.v_end):
            assert v in (v1 for c in self.cells(v) for v1 in self.vertices(c))


class SmoothingOpVeeserZanotti(SmoothingOpBase):

    def __new__(cls, V):
        if V.ufl_element().family() == 'Crouzeix-Raviart':
            return SmoothingOpVeeserZanottiCR.__new__(SmoothingOpVeeserZanottiCR, V)
        elif V.ufl_element().family() == 'Discontinuous Lagrange':
            return SmoothingOpVeeserZanottiDG.__new__(SmoothingOpVeeserZanottiDG, V)
        else:
            return object.__new__(cls)

    def __init__(self, V):
        assert V.ufl_element().degree() == 1
        self.V = V
        super(SmoothingOpVeeserZanotti, self).__init__(V.ufl_domain())

    @utils.cached_property
    def spaces(self):
        mesh = self.V.ufl_domain()
        P1 = FunctionSpace(mesh, 'P', 1)
        FB = FunctionSpace(mesh, 'FB', mesh.topological_dimension())
        return P1, FB


class SmoothingOpVeeserZanottiCR(SmoothingOpVeeserZanotti):

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, V):
        assert V.ufl_element().family() == 'Crouzeix-Raviart'
        super(SmoothingOpVeeserZanottiCR, self).__init__(V)

    @utils.cached_property
    def index_sets(self):
        f_start, f_end = self.f_start, self.f_end
        v_start, v_end = self.v_start, self.v_end
        facets, cone, first_cell = self.facets, self.cone, self.first_cell
        P1, FB = self.spaces

        # Compute facets in the "first cell" of each vertex
        F12 = [[f for f in facets(first_cell(v)) if v in cone(f)] for v in range(v_start, v_end)]
        F3 = [[f for f in facets(first_cell(v)) if v not in cone(f)] for v in range(v_start, v_end)]
        F1 = [ff[0] for ff in F12]  # first adjacent facet
        F2 = [ff[1] for ff in F12]  # second adjacent facet
        F3 = [ff[0] for ff in F3]   # opposite facet

        # Compose F1, F2, F3 with v0(f), v1(f), where v0, v1 give
        # vertices of a given facet
        FF11 = [F1[cone(f)[0]-v_start] for f in range(f_start, f_end)]
        FF12 = [F2[cone(f)[0]-v_start] for f in range(f_start, f_end)]
        FF13 = [F3[cone(f)[0]-v_start] for f in range(f_start, f_end)]
        FF21 = [F1[cone(f)[1]-v_start] for f in range(f_start, f_end)]
        FF22 = [F2[cone(f)[1]-v_start] for f in range(f_start, f_end)]
        FF23 = [F3[cone(f)[1]-v_start] for f in range(f_start, f_end)]

        # Map facet indices to CR dofs
        map_vals = np.vectorize(self.V.dm.getSection().getOffset, otypes=[utils.IntType])
        F1 = map_vals(F1)
        F2 = map_vals(F2)
        F3 = map_vals(F3)
        FF11 = map_vals(FF11)
        FF12 = map_vals(FF12)
        FF13 = map_vals(FF13)
        FF21 = map_vals(FF21)
        FF22 = map_vals(FF22)
        FF23 = map_vals(FF23)

        # Map vertex indices to P1 dofs
        map_inds = np.vectorize(P1.dm.getSection().getOffset, otypes=[utils.IntType])
        perm = map_inds(np.arange(v_start, v_end, dtype=utils.IntType))
        perm = np.argsort(perm)
        F1 = F1[perm]
        F2 = F2[perm]
        F3 = F3[perm]

        # Map facet indices to FB dofs
        map_inds = np.vectorize(FB.dm.getSection().getOffset, otypes=[utils.IntType])
        perm = map_inds(np.arange(f_start, f_end, dtype=utils.IntType))
        perm = np.argsort(perm)
        FF11 = FF11[perm]
        FF12 = FF12[perm]
        FF13 = FF13[perm]
        FF21 = FF21[perm]
        FF22 = FF22[perm]
        FF23 = FF23[perm]

        F1 = PETSc.IS().createGeneral(F1)
        F2 = PETSc.IS().createGeneral(F2)
        F3 = PETSc.IS().createGeneral(F3)
        FF11 = PETSc.IS().createGeneral(FF11)
        FF12 = PETSc.IS().createGeneral(FF12)
        FF13 = PETSc.IS().createGeneral(FF13)
        FF21 = PETSc.IS().createGeneral(FF21)
        FF22 = PETSc.IS().createGeneral(FF22)
        FF23 = PETSc.IS().createGeneral(FF23)

        return F1, F2, F3, FF11, FF12, FF13, FF21, FF22, FF23


    def apply(self, rhs, result=None):

        # Assemble the right-hand side on P1 and facet bubbles
        P1, FB = self.spaces
        f1 = getattr(self, 'f1', None)
        f2 = getattr(self, 'f2', None)
        self.f1 = f1 = assemble(rhs(TestFunction(P1)), tensor=f1).vector()
        self.f2 = f2 = assemble(rhs(TestFunction(FB)), tensor=f2).vector()

        # The new right-hand side shall be a functional (which is
        # conventionally handled as a Function in Firedrake) on V
        if result is None:
            result = Function(self.V)
        else:
            result.dat.zero()

        F1, F2, F3, FF11, FF12, FF13, FF21, FF22, FF23 = self.index_sets
        with result.dat.vec_wo as v:
            with f2.dat.vec_ro as v2:
                v.axpy(3/2, v2)
                vecisaxpy(v, FF11, -3/4, v2)
                vecisaxpy(v, FF12, -3/4, v2)
                vecisaxpy(v, FF13, +3/4, v2)
                vecisaxpy(v, FF21, -3/4, v2)
                vecisaxpy(v, FF22, -3/4, v2)
                vecisaxpy(v, FF23, +3/4, v2)
            with f1.dat.vec_ro as v1:
                vecisaxpy(v, F1, +1, v1)
                vecisaxpy(v, F2, +1, v1)
                vecisaxpy(v, F3, -1, v1)

        return result


class SmoothingOpVeeserZanottiDG(SmoothingOpVeeserZanotti):

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, V):
        assert V.ufl_element().family() == 'Discontinuous Lagrange'
        super(SmoothingOpVeeserZanottiDG, self).__init__(V)

    @utils.cached_property
    def index_sets(self):
        raise NotImplementedError

    def apply(self, rhs, result=None):
        raise NotImplementedError


def vecisaxpy(vfull, iset, alpha, vreduced):
    """Work around the bug in VecISAXPY:
    https://gitlab.com/petsc/petsc/-/issues/1357
    """
    if vfull.size == vreduced.size:
        for i, j in enumerate(iset.array):
            vfull[j] += alpha*vreduced[i]
    else:
        vfull.isaxpy(iset, alpha, vreduced)


if __name__ == '__main__':

    mesh = UnitSquareMesh(3, 3)
    for space in ['CR', 'DG']:
        V = FunctionSpace(mesh, space, 1)
        op = SmoothingOpVeeserZanotti(V)
        op.test_vertex_cell_conn()

        def rhs1(test_function):
            f = 1
            return f*test_function*dx
        F = op.apply(rhs1)

        def rhs2(test_function):
            f = as_vector([1, -1])
            return inner(f, grad(test_function))*dx
        op.apply(rhs2, result=F)
