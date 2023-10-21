import firedrake as fd
import matplotlib.pyplot as plt
import os


msh = fd.Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square2.msh")

fig, ax = plt.subplots()
fd.triplot(msh, axes=ax, boundary_kw={'colors': 'k'})
ax.set_aspect('equal')
plt.grid(False)
plt.axis('off')
plt.savefig('square2.pdf', transparent=True, bbox_inches="tight")
plt.show()
