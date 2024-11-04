#!/usr/bin/env python

# SBATCH -J RegressLoop
# SBATCH -p shared
# SBATCH -N 1
# SBATCH -n 1
# SBATCH -c 4
# SBATCH --mem 32GB
# SBATCH -t 0-2 # hours
# SBATCH -o job-scripts/regress/out/Job.%x.%A_%a.%N.%j.out # STDOUT
# SBATCH -e job-scripts/regress/err/Job.%x.%A_%a.%N.%j.err # STDERR
# SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
# SBATCH --mail-user=alowet@g.harvard.edu # send-to address

"""
Usage: sbatch loop_regress.sh

Spawn a new, multithreaded job for each neuron to fit with cvglmnet

Written by Adam S. Lowet, Jan. 16, 2021
"""

import os, sys

sys.path.append('../utils')
from paths import check_dir
from db import select_db, get_db_info, execute_sql
from regress_session import regress_session

RERUN = 0  # do we want to rerun analyses that have already been performed? (presumably with different parameters)

paths = get_db_info()
rets_im = execute_sql('SELECT * FROM imaging LEFT JOIN session ON imaging.behavior_path = session.raw_data_path WHERE session.has_facemap=1', paths['db'])
rets_ephys = execute_sql('SELECT * FROM ephys LEFT JOIN session ON ephys.behavior_path = session.raw_data_path WHERE session.has_facemap=1', paths['db'])
rets = rets_im + rets_ephys

job_directory = "./job-scripts/regress"

# Make top level directories
# mkdir_p(job_directory)
check_dir(os.path.join(job_directory, 'out'))
check_dir(os.path.join(job_directory, 'err'))

# Run a separate job for each neuron that has not been analyzed already
for ret in rets:
  print(ret)
  if not os.path.isfile(os.path.join(paths['neural_fig_roots'][0], ret['name'], ret['file_date_id'], 'regress_all.p')):
    print('Running regress_session on mouse {} file_date_id {}'.format(ret['name'], ret['file_date_id']))
    regress_session(ret['name'], ret['file_date'], ret['file_date_id'])
  else:
    print('regress_all.p found. Skipping mouse {} file_date_id {}'.format(ret['name'], ret['file_date_id']))

  for i_cell in range(ret['ncells']):
    check = select_db(paths['db'], 'neuron_regress', '*', 'mid=? AND file_date_id=? AND i_cell=?', (ret['mid'], ret['file_date_id'], i_cell), unique=False)
    if check and not RERUN:
      print('Already found entry for mouse {}, file_date_id {}, cell {}. Skipping'.format(ret['name'], ret['file_date_id'], i_cell))
      continue
    else:
      print('Running cvglmnet on mouse {}, file_date_id {}, cell {}'.format(ret['name'], ret['file_date_id'], i_cell))

      cell_name = '_'.join([ret['name'], ret['file_date_id'], 'cell{}'.format(i_cell)])
      job_file = os.path.join(job_directory, cell_name + '.job')

      with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s\n" % cell_name)
        fh.writelines("#SBATCH --output=%s.out\n" % os.path.join(job_directory, 'out', '%x.%A_%a.%N.%j'))
        fh.writelines("#SBATCH --error=%s.err\n" % os.path.join(job_directory, 'err', '%x.%A_%a.%N.%j'))
        fh.writelines("#SBATCH --time=0-8\n")  # minutes
        fh.writelines("#SBATCH --mem=32G\n")  # memory for each task
        fh.writelines("#SBATCH -p shared\n")  # partition (queue)
        fh.writelines("#SBATCH -N 1\n")  # nodes
        fh.writelines("#SBATCH -c 4\n")  # cpus
        fh.writelines("#SBATCH --mail-type=END,FAIL,TIME_LIMIT\n")
        fh.writelines("#SBATCH --mail-user=alowet@g.harvard.edu\n")
        fh.writelines("#SBATCH --export=ALL\n")

        # fh.writelines("module load Anaconda3/5.0.1-fasrc02\n")
        fh.writelines("module load python/3.10.9-fasrc01\n")
        fh.writelines("source activate jupyter37\n")
        fh.writelines("python3 -u regress_cell.py %s %s %s %s\n" % (ret['name'], ret['file_date'], ret['file_date_id'], i_cell))
        fh.writelines("echo regress_cell complete on file %s\n" % cell_name)
      os.system("sbatch %s" % job_file)
