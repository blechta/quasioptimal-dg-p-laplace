"""Script to figure out how IGNORE_NEGITIVE_INDICES work.

If ON, negative indices are indeed ignored.
If OFF and PETSc is built in DEBUG mode, an error is raised.
If OFF and PETSc not in DEBUG mode, a negative index really causes a write into
the memory (ahead of the array start), possibly leading to memory corruption
and a segfault.

THe tricky part is interpreting the case with the flag ON: v[-1] = 10 indeed
makes no write, but a subsequent use of the expression v[-1] apparently revives
the previously allocated rhs array thus seemingly recovering the "unwritten"
value 10.
"""

from petsc4py import PETSc

v = PETSc.Vec()
v.createSeq(10)
v.setOption(PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)
v.view()
print(v[-1])
v[-1] = 10
print(v[-1])

v.setOption(PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, False)
for i in range(-1, -1000, -1):
    v[i] = 42  # This will cause memory corruption around i = -137
