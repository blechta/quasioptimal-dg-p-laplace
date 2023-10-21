#!/bin/bash
set -e

# Select image
IMAGE="firedrakeproject/firedrake:2023-10"

# PUBLICATION TITLE
TITLE='Quasi-optimal Discontinuous Galerkin discretisations of the $p$-Dirichlet equation'

# Compose Docker command to be run on the host
DOCKER_RUN_CMD=(
    docker
    run
    --rm
    ${IMAGE}
    /bin/bash
    -c
)

# Compose commands to be run in the container
read -r -d '' CONTAINER_CMD <<EOF || true
set -e

# Activate Firedrake virtualenv
source ~/firedrake/bin/activate

# Symlink PETSc and SLEPc to expected location
ln -s /home/firedrake/petsc /home/firedrake/firedrake/src/
ln -s /home/firedrake/slepc /home/firedrake/firedrake/src/
ln -s /home/firedrake/firedrake/src/petsc/src/binding/petsc4py /home/firedrake/firedrake/src/
ln -s /home/firedrake/firedrake/src/slepc/src/binding/slepc4py /home/firedrake/firedrake/src/

# Recover deleted .out files
cd /home/firedrake/firedrake/src/petsc
git reset --hard

components=(firedrake PyOP2 tsfc ufl FInAT fiat petsc petsc4py loopy slepc slepc4py)
for comp in "\${components[@]}"; do
    cd /home/firedrake/firedrake/src/\${comp}
    # Fix stale stat cache: https://stackoverflow.com/a/36439778
    git update-index --refresh
done

# Release
cd /tmp
firedrake-zenodo -t '${TITLE}'
cat firedrake.json
EOF

# Run actual commands
"${DOCKER_RUN_CMD[@]}" "${CONTAINER_CMD}"
