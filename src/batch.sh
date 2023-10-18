#!/bin/bash
#
#SBATCH --job-name=paper-quasioptimal
#SBATCH --output=paper-quasioptimal.log
#
#SBATCH -n 18
#SBATCH --time=12:00:00
#SBATCH -p express3
#SBATCH -w r32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=blechta@karlin.mff.cuni.cz

cd /usr/work/blechta/dev/paper-smoothing-op/src
udocker run -v $PWD:/mnt quasioptimal bash /mnt/script.sh
