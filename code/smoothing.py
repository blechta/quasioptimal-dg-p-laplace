from firedrake import *
from petsc4py import PETSc
import numpy as np


class SmoothingOpBase(object):

    def __init__(self, mesh):
        assert mesh.topological_dimension() == 2, '3d not yet implemented'
        self.mesh = mesh
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

    def bc_facets(self):
        return self.mesh.exterior_facets.facets

    def bc_vertices(self):
        bc_facets = self.bc_facets()
        bc_vertices = {v for f in bc_facets for v in self.cone(f)}
        num_bc_vertices = len(bc_vertices)
        bc_vertices = np.fromiter(bc_vertices, dtype=utils.IntType)
        bc_vertices = np.unique(bc_vertices)
        assert bc_vertices.size == num_bc_vertices
        return bc_vertices


class SmoothingOpVeeserZanotti(SmoothingOpBase):

    def __new__(cls, V):
        if V.ufl_element().family() == 'Crouzeix-Raviart':
            return SmoothingOpVeeserZanottiCR(V)
        elif V.ufl_element().family() == 'Discontinuous Lagrange':
            return SmoothingOpVeeserZanottiDG(V)
        else:
            raise NotImplementedError(f'Smoothing not implemented for space {V}')

    def __init__(self, V):
        assert V.ufl_element().degree() == 1
        self.V = V
        super().__init__(V.ufl_domain())

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
                    vecisaxpy(v, iset, alpha, v1)
            with f2.dat.vec_ro as v2:
                for iset, alpha in coeffs2:
                    vecisaxpy(v, iset, alpha, v2)

        return result

    def permute_vertex_indexed_to_p1(self, *args):
        P1, _ = self.spaces
        map_inds = np.vectorize(P1.dm.getSection().getOffset, otypes=[utils.IntType])
        perm = map_inds(np.arange(self.v_start, self.v_end, dtype=utils.IntType))
        perm = np.argsort(perm)
        return self.to_is(*(arr[perm] for arr in args))

    def permute_facet_indexed_to_fb(self, *args):
        _, FB = self.spaces
        map_inds = np.vectorize(FB.dm.getSection().getOffset, otypes=[utils.IntType])
        perm = map_inds(np.arange(self.f_start, self.f_end, dtype=utils.IntType))
        perm = np.argsort(perm)
        return self.to_is(*(arr[perm] for arr in args))

    @staticmethod
    def to_is(*args):
        return [PETSc.IS().createGeneral(arr) for arr in args]


class SmoothingOpVeeserZanottiCR(SmoothingOpVeeserZanotti):

    def __new__(cls, V, *args, **kwargs):
        assert V.ufl_element().family() == 'Crouzeix-Raviart'
        return object.__new__(cls)

    @utils.cached_property
    def coeffs(self):
        f_start, f_end = self.f_start, self.f_end
        v_start, v_end = self.v_start, self.v_end
        facets, cone, first_cell = self.facets, self.cone, self.first_cell

        # Compute boundary entities
        bc_facets = self.bc_facets()
        bc_vertices = self.bc_vertices()

        # Compute facets in the "first cell" of each vertex
        F12 = [[f for f in facets(first_cell(v)) if v in cone(f)] for v in range(v_start, v_end)]
        F3 = [[f for f in facets(first_cell(v)) if v not in cone(f)] for v in range(v_start, v_end)]
        F12 = np.array(F12, dtype=utils.IntType)
        F3 = np.array(F3, dtype=utils.IntType)
        F1 = F12[:, 0]  # first adjacent facet
        F2 = F12[:, 1]  # second adjacent facet
        F3 = F3[:, 0]   # opposite facet

        # Part that is identity modulo BC
        FF0 = np.arange(f_start, f_end, dtype=utils.IntType)

        # Translate facet indices to CR dofs and apply BC
        F1, F2, F3, FF0 = self.facets_to_cr_dofs(F1, F2, F3, FF0)
        F1[bc_vertices-v_start] = -1
        F2[bc_vertices-v_start] = -1
        F3[bc_vertices-v_start] = -1
        FF0[bc_facets-f_start] = -1

        # Compose F1, F2, F3 with v0(f), v1(f), where v0, v1 give
        # vertices of a given facet
        facet_vertices = [cone(f) for f in range(f_start, f_end)]
        facet_vertices = np.array(facet_vertices, dtype=utils.IntType)
        FF11 = F1[facet_vertices[:, 0]-v_start]
        FF12 = F2[facet_vertices[:, 0]-v_start]
        FF13 = F3[facet_vertices[:, 0]-v_start]
        FF21 = F1[facet_vertices[:, 1]-v_start]
        FF22 = F2[facet_vertices[:, 1]-v_start]
        FF23 = F3[facet_vertices[:, 1]-v_start]
        FF11[bc_facets-f_start] = -1
        FF12[bc_facets-f_start] = -1
        FF13[bc_facets-f_start] = -1
        FF21[bc_facets-f_start] = -1
        FF22[bc_facets-f_start] = -1
        FF23[bc_facets-f_start] = -1

        # Map mesh entity indices to dofs of corresponding spaces
        F1, F2, F3 = self.permute_vertex_indexed_to_p1(F1, F2, F3)
        FF0, FF11, FF12, FF13, FF21, FF22, FF23 = \
            self.permute_facet_indexed_to_fb(FF0, FF11, FF12, FF13, FF21, FF22, FF23)

        coeffs1 = [(F1, +1), (F2, +1), (F3, -1)]
        coeffs2 = [(FF0, 3/2),
                   (FF11, -3/4), (FF12, -3/4), (FF13, +3/4),
                   (FF21, -3/4), (FF22, -3/4), (FF23, +3/4)]
        return coeffs1, coeffs2

    def facets_to_cr_dofs(self, *args):
        CR = self.V
        map_vals = np.vectorize(CR.dm.getSection().getOffset, otypes=[utils.IntType])
        return (map_vals(arr) for arr in args)


class SmoothingOpVeeserZanottiDG(SmoothingOpVeeserZanotti):

    def __new__(cls, V, *args, **kwargs):
        assert V.ufl_element().family() == 'Discontinuous Lagrange'
        return object.__new__(cls)

    @utils.cached_property
    def coeffs(self):
        c_start = self.c_start
        f_start, f_end = self.f_start, self.f_end
        v_start, v_end = self.v_start, self.v_end
        facet_cells, cone, first_cell = self.facet_cells, self.cone, self.first_cell
        (P1, FB), DG1 = self.spaces, self.V

        # Compute boundary entities
        bc_facets = self.bc_facets()
        bc_vertices = self.bc_vertices()

        # Construct mapping from cell and vertex to DG1 dof
        f2p = DG1.mesh().cell_closure[:, -1]
        p2f = np.argsort(f2p)
        assert c_start == 0
        fdofs = DG1.cell_node_list
        def map_vals_(pc, lv):
            if pc < 0 or lv < 0:
                return -1
            return fdofs[p2f[pc], lv]
        map_vals = np.vectorize(map_vals_, otypes=[utils.IntType])

        # Compute "first cell" of each vertex
        Kv = [first_cell(v) for v in range(v_start, v_end)]

        # Compute local vertex number for each v in Kv
        fvertices = DG1.mesh().cell_closure[:, 0:3]
        Kvertices = np.array([fvertices[p2f[c]] for c in Kv])
        mask = Kvertices.T == np.arange(v_start, v_end)
        local_v_in_Kv = mask.argmax(axis=0)

        # Map cell and local vertex index pairs to DG1 dofs
        F = map_vals(Kv, local_v_in_Kv)
        F[bc_vertices-v_start] = -1

        # Compose F with v0(f), v1(f), where v0, v1 give
        # vertices of a given facet
        FF1 = [F[cone(f)[0]-v_start] for f in range(f_start, f_end)]
        FF2 = [F[cone(f)[1]-v_start] for f in range(f_start, f_end)]
        FF1 = np.array(FF1, dtype=utils.IntType)
        FF2 = np.array(FF2, dtype=utils.IntType)
        FF1[bc_facets-f_start] = -1
        FF2[bc_facets-f_start] = -1

        # Compute cells and vertices for all facets
        C1 = [facet_cells(f)[0] for f in range(f_start, f_end)]
        C2 = [facet_cells(f)[1] for f in range(f_start, f_end)]  # -1 if facet on boundary
        V1 = [cone(f)[0] for f in range(f_start, f_end)]
        V2 = [cone(f)[1] for f in range(f_start, f_end)]

        # Compute local vertex number for each vertex and cell
        C1vertices = np.array([fvertices[p2f[c]] for c in C1])
        C2vertices = np.array([fvertices[p2f[c]] if c>-1 else [-1,-1,-1] for c in C2])
        mask11 = C1vertices.T == np.array(V1)
        mask12 = C1vertices.T == np.array(V2)
        mask21 = C2vertices.T == np.array(V1)
        mask22 = C2vertices.T == np.array(V2)
        local_V1_in_C1 = mask11.argmax(axis=0)
        local_V2_in_C1 = mask12.argmax(axis=0)
        local_V1_in_C2 = np.where(mask21.any(axis=0), mask21.argmax(axis=0), -1)
        local_V2_in_C2 = np.where(mask22.any(axis=0), mask22.argmax(axis=0), -1)

        # Map cell and local vertex index pairs to DG1 dofs
        FF11 = map_vals(C1, local_V1_in_C1)
        FF12 = map_vals(C1, local_V2_in_C1)
        FF21 = map_vals(C2, local_V1_in_C2)  # -1 if facet on boundary
        FF22 = map_vals(C2, local_V2_in_C2)  # -1 if facet on boundary
        FF11[bc_facets-f_start] = -1
        FF12[bc_facets-f_start] = -1
        FF21[bc_facets-f_start] = -1
        FF22[bc_facets-f_start] = -1

        # Map mesh entity indices to dofs of corresponding spaces
        F, = self.permute_vertex_indexed_to_p1(F)
        FF1, FF2, FF11, FF12, FF21, FF22 = \
            self.permute_facet_indexed_to_fb(FF1, FF2, FF11, FF12, FF21, FF22)

        coeffs1 = [(F, +1)]
        coeffs2 = [(FF1, -3/4), (FF2, -3/4),
                   (FF11, 3/8), (FF12, 3/8), (FF21, 3/8), (FF22, 3/8)]
        return coeffs1, coeffs2


def vecisaxpy(vfull, iset, alpha, vreduced):
    """Work around the bug in VecISAXPY:
    https://gitlab.com/petsc/petsc/-/issues/1357

    TODO: Fixed in PETSc >= 3.19.1; see
    https://gitlab.com/petsc/petsc/-/commit/02a61dc80d8db8beafc07ae7f4b29a54f31ade06
    Bump PETSc version and remove workaround.
    """
    if vfull.size == vreduced.size:
        rstart, rend = vfull.owner_range
        for i, j in enumerate(iset.array):
            if j < 0:
                continue
            if j >= rend or j < rstart:
                raise ValueError
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
