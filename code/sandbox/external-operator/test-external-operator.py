from ufl import *
from ufl.domain import affine_mesh

cell = tetrahedron
k = 4

mesh = affine_mesh(cell)
DG = FunctionSpace(mesh, FiniteElement("DG", cell, k))
P = FunctionSpace(mesh, FiniteElement("P", cell, k+1))

u = TrialFunction(DG)
v = TestFunction(DG)
w = ExternalOperator(v, function_space=P)

L = w*dx
a = inner(grad(u), grad(w))*dx
# + other DG terms tested with smooth(v)

# Test
arg0, = L.arguments()
assert arg0 == v
arg0, arg1 = a.arguments()
assert arg0 == v and arg1 == u
