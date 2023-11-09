#!/bin/bash
#
#SBATCH --job-name paper-quasioptimal
#SBATCH --output paper-quasioptimal.log
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --time=12:00:00
#SBATCH --partition express3
#SBATCH --nodelist r40
#SBATCH --mail-type ALL
#SBATCH --mail-user blechta@karlin.mff.cuni.cz

set -e

NPROC=$SLURM_CPUS_PER_TASK
CONTAINER=quasioptimal
CMD="source ~/firedrake/bin/activate; make -j $NPROC"

udocker run -v "$PWD:/mnt" -w /mnt $CONTAINER /bin/bash -c "$CMD"
