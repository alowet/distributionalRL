#!/bin/bash

#SBATCH -J SpikeLoop
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 4:00:00
#SBATCH -o job-scripts/out/Job.%x.%A_%a.%N.%j.out # STDOUT
#SBATCH -e job-scripts/err/Job.%x.%A_%a.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
#SBATCH --mail-user=alowet@g.harvard.edu # send-to address

# Usage: loop_spikes.sh
# Written by Adam S. Lowet, Dec. 15, 2020

# module load Anaconda3/5.0.1-fasrc02
module load python/3.10.9-fasrc01
source activate neural
srun -N 1 -c $SLURM_CPUS_PER_TASK python3 -u loop_spikes.py