#!/bin/bash
source ~/firedrake/bin/activate
cd /mnt/examples/
mkdir -p output/primal/ output/mixed/
gmsh -2 -clscale 0.36 square_str.geo
make -j 16
