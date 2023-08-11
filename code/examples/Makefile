all:
	gmsh -2 -clscale 0.3 square.geo -smooth 3
	gmsh -3 -clscale 1.0 cube.geo -algo del3d -optimize_netgen 5 -smooth 10
