#!/bin/bash

#SBATCH -J BehaviorLoop
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem 16G
#SBATCH -t 1:00:00
#SBATCH -o job-scripts/out/Job.%x.%A_%a.%N.%j.out # STDOUT
#SBATCH -e job-scripts/err/Job.%x.%A_%a.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
#SBATCH --mail-user=alowet@g.harvard.edu # send-to address

# Usage: sbatch loop_behavior.sh
# Written by Adam S. Lowet, Nov. 20, 2019

# module load Anaconda3/5.0.1-fasrc02
module load python/3.10.9-fasrc01
source activate behavior
srun -N 1 -c $SLURM_CPUS_PER_TASK python3 -u loop_behavior_db.py