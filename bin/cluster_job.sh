#!/bin/bash
#
#SBATCH --job-name paper-quasioptimal
#SBATCH --output paper-quasioptimal.log
#SBATCH --ntasks 1
#SBATCH --nodelist r40
#SBATCH --cpus-per-task 144
#SBATCH --partition express3
#SBATCH --time=12:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user blechta@karlin.mff.cuni.cz

set -e

NPROC=12
CONTAINER=quasioptimal
CMD="source ~/firedrake/bin/activate; make -j $NPROC"

udocker run -v "$PWD:/mnt" -w /mnt $CONTAINER /bin/bash -c "$CMD"
