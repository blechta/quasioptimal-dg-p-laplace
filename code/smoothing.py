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

    def facet_cells(self, facet):
        assert facet >= self.f_start and facet < self.f_end
        cells = [c for c in self.dm.getAdjacency(facet)
                 if c >= self.c_start and c < self.c_end]
        if len(cells)==1:
            cells.append(-1)
        c1, c2 = cells
        return c1, c2

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
            #return SmoothingOpVeeserZanottiCR(V)
        elif V.ufl_element().family() == 'Discontinuous Lagrange':
            return SmoothingOpVeeserZanottiDG.__new__(SmoothingOpVeeserZanottiDG, V)
            #return SmoothingOpVeeserZanottiDG(V)
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

    def assemble_rhs(self, rhs):
        # Assemble the right-hand side on P1 and facet bubbles
        P1, FB = self.spaces
        f1 = getattr(self, 'f1', None)
        f2 = getattr(self, 'f2', None)
        self.f1 = f1 = assemble(rhs(TestFunction(P1)), tensor=f1).vector()
        self.f2 = f2 = assemble(rhs(TestFunction(FB)), tensor=f2).vector()
        return f1, f2

    def apply(self, rhs, result=None):

        # The new right-hand side shall be a functional (which is
        # conventionally handled as a Function in Firedrake) on V
        if result is None:
            result = Function(self.V)
        else:
            result.dat.zero()

        f1, f2 = self.assemble_rhs(rhs)
        coeffs1, coeffs2 = self.coeffs
        with result.dat.vec_wo as v:
            with f1.dat.vec_ro as v1:
                for iset, alpha in coeffs1:
                    isaxpy_or_axpy(v, iset, alpha, v1)
            with f2.dat.vec_ro as v2:
                for iset, alpha in coeffs2:
                    isaxpy_or_axpy(v, iset, alpha, v2)

        return result


class SmoothingOpVeeserZanottiCR(SmoothingOpVeeserZanotti):

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, V):
        assert V.ufl_element().family() == 'Crouzeix-Raviart'
        super(SmoothingOpVeeserZanottiCR, self).__init__(V)

    @utils.cached_property
    def coeffs(self):
        f_start, f_end = self.f_start, self.f_end
        v_start, v_end = self.v_start, self.v_end
        facets, cone, first_cell = self.facets, self.cone, self.first_cell
        (P1, FB), CR = self.spaces, self.V

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
        map_vals = np.vectorize(CR.dm.getSection().getOffset, otypes=[utils.IntType])
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

        coeffs1 = [(F1, +1), (F2, +1), (F3, -1)]
        coeffs2 = [(None, 3/2),
                   (FF11, -3/4), (FF12, -3/4), (FF13, +3/4),
                   (FF21, -3/4), (FF22, -3/4), (FF23, +3/4)]
        return coeffs1, coeffs2


class SmoothingOpVeeserZanottiDG(SmoothingOpVeeserZanotti):

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, V):
        assert V.ufl_element().family() == 'Discontinuous Lagrange'
        super(SmoothingOpVeeserZanottiDG, self).__init__(V)

    @utils.cached_property
    def coeffs(self):
        f_start, f_end = self.f_start, self.f_end
        v_start, v_end = self.v_start, self.v_end
        vertices, facet_cells, cone, first_cell = self.vertices, self.facet_cells, self.cone, self.first_cell
        (P1, FB), DG1 = self.spaces, self.V

        # Compute "first cell" of each vertex
        Kv = [first_cell(v) for v in range(v_start, v_end)]

        # Compute local vertex number for each v in Kv
        Kvertices = np.array([vertices(c) for c in Kv])
        mask = Kvertices.T == np.arange(v_start, v_end)
        local_v_in_Kv = mask.argmax(axis=0)

        # Map cell and local vertex index pairs to DG1 dofs
        f2p = DG1.mesh().cell_closure[:, -1]
        p2f = np.argsort(f2p)
        fdofs = DG1.cell_node_list
        def map_vals_(pc, lv):
            return fdofs[p2f[pc], lv]
        map_vals = np.vectorize(map_vals_, otypes=[utils.IntType])
        F = map_vals(Kv, local_v_in_Kv)

        # Compose F with v0(f), v1(f), where v0, v1 give
        # vertices of a given facet
        # FIXME: Maybe we're ignoring BC here
        FF1 = [F[cone(f)[0]-v_start] for f in range(f_start, f_end)]
        FF2 = [F[cone(f)[1]-v_start] for f in range(f_start, f_end)]

        FF1 = np.array(FF1)
        FF2 = np.array(FF2)

        # Compute cells and vertices for all facets
        C1 = [facet_cells(f)[0] for f in range(f_start, f_end)]
        C2 = [facet_cells(f)[1] for f in range(f_start, f_end)]  # -1 if facet on boundary
        V1 = [cone(f)[0] for f in range(f_start, f_end)]
        V2 = [cone(f)[1] for f in range(f_start, f_end)]

        # Compute local vertex number for each vertex and cell
        C1vertices = np.array([vertices(c) for c in C1])
        C2vertices = np.array([vertices(c) if c>-1 else [-1,-1,-1] for c in C2])
        mask11 = C1vertices.T == np.array(V1)
        mask12 = C1vertices.T == np.array(V2)
        mask21 = C2vertices.T == np.array(V1)  # FIXME: How to handle -1's?
        mask22 = C2vertices.T == np.array(V2)  # FIXME: How to handle -1's?
        local_V1_in_C1 = mask11.argmax(axis=0)
        local_V2_in_C1 = mask12.argmax(axis=0)
        local_V1_in_C2 = mask21.argmax(axis=0)  # FIXME: How to handle -1's?
        local_V2_in_C2 = mask22.argmax(axis=0)  # FIXME: How to handle -1's?

        # Map cell and local vertex index pairs to DG1 dofs
        FF11 = map_vals(C1, local_V1_in_C1)
        FF12 = map_vals(C1, local_V2_in_C1)
        FF21 = map_vals(C2, local_V1_in_C2)  # FIXME: How to handle -1's?
        FF22 = map_vals(C2, local_V2_in_C2)  # FIXME: How to handle -1's?

        # Map vertex indices to P1 dofs
        map_inds = np.vectorize(P1.dm.getSection().getOffset, otypes=[utils.IntType])
        perm = map_inds(np.arange(v_start, v_end, dtype=utils.IntType))
        perm = np.argsort(perm)
        F = F[perm]

        # Map facet indices to FB dofs
        map_inds = np.vectorize(FB.dm.getSection().getOffset, otypes=[utils.IntType])
        perm = map_inds(np.arange(f_start, f_end, dtype=utils.IntType))
        perm = np.argsort(perm)
        FF1 = FF1[perm]
        FF2 = FF2[perm]
        FF11 = FF11[perm]
        FF12 = FF12[perm]
        FF21 = FF21[perm]
        FF22 = FF22[perm]

        F = PETSc.IS().createGeneral(F)
        FF1 = PETSc.IS().createGeneral(FF1)
        FF2 = PETSc.IS().createGeneral(FF2)
        FF11 = PETSc.IS().createGeneral(FF11)
        FF12 = PETSc.IS().createGeneral(FF12)
        FF21 = PETSc.IS().createGeneral(FF21)
        FF22 = PETSc.IS().createGeneral(FF22)

        coeffs1 = [(F, +1)]
        coeffs2 = [(FF1, -3/4), (FF2, -3/4),
                   (FF11, 3/8), (FF12, 3/8), (FF21, 3/8), (FF22, 3/8)]
        #import pdb; pdb.set_trace()
        return coeffs1, coeffs2

    def assemble_rhs(self, rhs):
        f1, f2 = super(SmoothingOpVeeserZanottiDG, self).assemble_rhs(rhs)
        bc1, bc2 = self.bcs_rhs
        bc1.apply(f1)
        bc2.apply(f2)
        return f1, f2

    @utils.cached_property
    def bcs_rhs(self):
        P1, FB = self.spaces
        bc1 = DirichletBC(P1, Constant(0.), "on_boundary")
        bc2 = DirichletBC(FB, Constant(0.), "on_boundary")
        return bc1, bc2


def isaxpy_or_axpy(vfull, iset, alpha, vreduced):
    """Perform VecISAXPY or VecAXPY if iset is None"""
    if iset is None:
        vfull.axpy(alpha, vreduced)
    else:
        vecisaxpy(vfull, iset, alpha, vreduced)


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
