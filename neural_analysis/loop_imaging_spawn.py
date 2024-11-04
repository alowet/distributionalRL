# """
# Usage: sbatch loop_imaging_spawn.sh, which calls this script
#
# Spawn a new, single-threaded job for imaging session to plot
#
# Written by Adam S. Lowet, Oct. 9, 2023
# """
#
# import os, sys
#
# sys.path.append('../utils')
# from paths import check_dir
# from db import select_db, get_db_info, execute_sql
# #from plotTrialAvgsByNeuron import plotTrialAvgsByNeuron
#
# RERUN = 1  # do we want to rerun analyses that have already been performed? (presumably with different parameters)
#
# paths = get_db_info()
# if RERUN:
#   rets = execute_sql('SELECT * FROM imaging WHERE curated=1 AND name>="AL60" AND name<"AL68"', paths['db'])
# else:
#   rets = execute_sql('SELECT * FROM imaging WHERE date_processed IS NULL AND curated=1', paths['db'])
#
# job_directory = "./job-scripts/image"
#
# # Make top level directories
# check_dir(os.path.join(job_directory, 'out'))
# check_dir(os.path.join(job_directory, 'err'))
#
# #print(rets)
#
# # Run a separate job for each neuron that has not been analyzed already
# for ret in rets:
#   print(ret)
#   job_name = f"{ret['name']}_{ret['file_date_id']}"
#   job_file = os.path.join(job_directory, f"{job_name}.job")
#
#   with open(job_file, 'w') as fh:
#     fh.writelines("#!/bin/bash\n")
#     fh.writelines("#SBATCH --job-name=%s\n" % job_name)
#     fh.writelines("#SBATCH --output=%s.out\n" % os.path.join(job_directory, 'out', '%x.%A_%a.%N.%j'))
#     fh.writelines("#SBATCH --error=%s.err\n" % os.path.join(job_directory, 'err', '%x.%A_%a.%N.%j'))
#     fh.writelines("#SBATCH --time=0-2\n")  # hours
#     fh.writelines("#SBATCH --mem=8G\n")  # memory for each task
#     fh.writelines("#SBATCH -p shared\n")  # partition (queue)
#     fh.writelines("#SBATCH -N 1\n")  # nodes
#     fh.writelines("#SBATCH -c 1\n")  # cpus
#     fh.writelines("#SBATCH --mail-type=END,FAIL,TIME_LIMIT\n")
#     fh.writelines("#SBATCH --mail-user=alowet@g.harvard.edu\n")
#     fh.writelines("#SBATCH --export=ALL\n")
#
#     fh.writelines("module load python/3.10.9-fasrc01\n")
#     fh.writelines("source activate suite2p\n")
#     fh.writelines("python3 -u plotTrialAvgsByNeuron.py %s\n" % os.path.join(paths['imaging_root'], ret['name'], ret['file_date_id']))
# #    fh.writelines("echo plotTrialAvgsByNeuron complete on file %s\n" % job_name)
#
#   os.system("sbatch %s" % job_file)
