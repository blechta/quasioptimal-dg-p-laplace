"""This scripts tests my knowledge how dof mapping
between PETSc and Firedrake etc. works
"""

from firedrake import *
import numpy as np

#nx = ny = 1
nx = 28
ny = 74

mesh = UnitSquareMesh(nx, ny)
space = 'DG'
V = FunctionSpace(mesh, space, 1)

fdofs = V.cell_node_list
perm = V.finat_element.entity_permutations[2][0]

v_start, v_end = mesh.topology_dm.getDepthStratum(0)
c_start, c_end = mesh.topology_dm.getHeightStratum(0)
f_start, f_end = mesh.topology_dm.getHeightStratum(1)

map_vals = np.vectorize(V.dm.getSection().getOffset, otypes=[utils.IntType])
offsets = map_vals(range(c_start, c_end))

f2p = mesh.cell_closure[:,-1]
p2f = np.argsort(f2p)

#print(offsets[f2p])

x, y = SpatialCoordinate(mesh)
f = Function(V).interpolate(x + y*1000.0)

for pc in range(c_start, c_end):
    #c_vertices = [v for v in mesh.topology_dm.getAdjacency(pc)
    #              if v >= v_start and v < v_end]
    #fc = p2f[pc]

    fc = p2f[pc]

    #for lv, pv in enumerate(c_vertices):
    for lv, pv in enumerate(mesh.cell_closure[fc, 0:3]):
        fdof = fdofs[fc, lv]

        fdof_coords = mesh.coordinates.function_space().dm.getSection().getOffset(pv)
        x, y = mesh.coordinates.dat.data_ro[fdof_coords]

        #print(f.dat.data_ro[fdof], x + 1000*y)
        assert np.isclose(f.dat.data_ro[fdof], x + 1000*y)
