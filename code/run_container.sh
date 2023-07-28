#!/bin/bash
set -e

# Select image
IMAGE="fd:paper-smoothing-op"  # FIXME: currently a custom firedrake tag

THISDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONTAINER_MOUNT_POINT="/home/firedrake/shared"

if [ -t 0 ] ; then
    # Interactive
    DOCKER_RUN="docker run --interactive --tty"
else
    # No input terminal
    DOCKER_RUN="docker run"
fi

read -r -d '' DOCKER_RUN_CMD <<EOF || true
${DOCKER_RUN}
  --rm \
  --volume=${THISDIR}:${CONTAINER_MOUNT_POINT} \
  --env=DISPLAY=${DISPLAY} \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix \
  --workdir=${CONTAINER_MOUNT_POINT} \
  ${IMAGE} \
  /bin/bash -c
EOF

read -r -d '' CONTAINER_CMD <<EOF || true
set -e
ln -s ${CONTAINER_MOUNT_POINT}/.{bash,python}_history ~/
source ~/firedrake/bin/activate
export OMP_NUM_THREADS=1
rm -rf \${VIRTUAL_ENV}/.cache
ln -s ${CONTAINER_MOUNT_POINT}/.cache \${VIRTUAL_ENV}/
mkdir -p ${CONTAINER_MOUNT_POINT}/.cache/tsfc/
#pip install --no-cache-dir pdbpp
set -x
"\$0" "\$@"
EOF

set -x
${DOCKER_RUN_CMD} "${CONTAINER_CMD}" "$@"
