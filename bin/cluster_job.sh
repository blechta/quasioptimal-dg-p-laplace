#!/bin/bash
#
#SBATCH --job-name=paper-quasioptimal
#SBATCH --output=paper-quasioptimal.log
#SBATCH -n 32
#SBATCH --time=12:00:00
#SBATCH -p express3
#SBATCH -w r40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=blechta@karlin.mff.cuni.cz

set -e

NPROC=30
CONTAINER=quasioptimal
CMD="source ~/firedrake/bin/activate; make -j $NPROC"

udocker run -v "$PWD:/mnt" -w /mnt $CONTAINER /bin/bash -c "$CMD"
