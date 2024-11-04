#!/n/holystore01/LABS/uchida_users/Users/alowet/envs/tf/bin/python

#SBATCH -J GPURegressLoop
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 16GB
#SBATCH -t 10 # minutes
#SBATCH -o job-scripts/gpu_regress/out/Job.%x.%A_%a.%N.%j.out # STDOUT
#SBATCH -e job-scripts/gpu_regress/err/Job.%x.%A_%a.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
#SBATCH --mail-user=alowet@g.harvard.edu # send-to address

"""
Usage: sbatch loop_regress_gpu.sh

Spawn a new GPU job for each session to fit with regress_session.py

Written by Adam S. Lowet, Feb. 14, 2024
"""

import os, sys

sys.path.append('../utils')
from paths import check_dir
from db import select_db, get_db_info, execute_sql
import random
import json

n_sessions_per_chunk = 6

# do we want to rerun analyses that have already been performed? (presumably with different parameters)
# elastic net is optimized with lr=1e-3, se_frac=0
#refit = 0; lr = 1e-3; reg = 'elastic_net'; se_frac = 0; l1_ratio = 0.0
# group_lasso is optimized with lr=5e-3, se_frac=0.75
refit = 0; lr = 5e-3; reg = 'group_lasso'; se_frac = 0.75; l1_ratio = 0.0

#n_sessions_per_chunk = 190  # just run one job to do evaluation. Fitting complete

table = 'imaging'
dt = 0.020 if table == 'ephys' else 1/15.24

paths = get_db_info()
rets = execute_sql(f'SELECT * FROM {table} LEFT JOIN session ON {table}.behavior_path = session.raw_data_path WHERE session.has_facemap=1', paths['db'])
print(len(rets))

#paths = get_db_info()
#rets_im = execute_sql('SELECT * FROM imaging LEFT JOIN session ON imaging.behavior_path = session.raw_data_path WHERE session.has_facemap=1', paths['db'])
#rets_ephys = execute_sql('SELECT * FROM ephys LEFT JOIN session ON ephys.behavior_path = session.raw_data_path WHERE session.has_facemap=1', paths['db'])
#rets = rets_im + rets_ephys

job_directory = "./job-scripts/gpu_regress"

# Make top level directories
# mkdir_p(job_directory)
check_dir(os.path.join(job_directory, 'out'))
check_dir(os.path.join(job_directory, 'err'))

# chunk sessions. Run a separate job per chunk
#random.Random(1).shuffle(rets)
#chunks = [rets[i:i+n_sessions_per_chunk] for i in range(0, len(rets), n_sessions_per_chunk)]
#print(len(chunks))
# print(chunks)

#for i_chunk, chunk in enumerate(chunks):
job_count = 0
while len(rets) > 0:
  to_analyze = []
#  if refit: to_analyze = chunk
#  else:
#  for ret in chunk:  # only analyze the session if it has not already been fit
  while len(to_analyze) < n_sessions_per_chunk and len(rets) > 0:
    ret = rets.pop(0)
    in_table = select_db('my', 'glm_fit', '*', 'mid=? AND sid=? AND rid=? AND se_frac=? AND regularization=? AND ' + \
      'learning_rate=? AND l1_ratio=? AND dt=? AND reward_dev_abl_nuissance IS NOT NULL',  # AND NOT (name="AL28" AND exp_date=20210415)',
      (ret['mid'], ret['sid'], ret['rid'], se_frac, reg, lr, l1_ratio, dt), unique=False)
#    print(len(in_table))
    if len(in_table) == 0:
      to_analyze.append(ret)
    else:
      print('Excluding ', ret['name'], ret['file_date'])

  job_count += 1
  if len(to_analyze) > 0: # and job_count > 1:

    prefix = '_'.join(['_'.join([ret['name'][1:], ret['file_date_id'][-5:]]) for ret in to_analyze]) + '_{:.2f}'.format(l1_ratio) if len(to_analyze) < 20 else 'all'
    if table == 'imaging':
      args = [(ret['name'], ret['file_date'], ret['file_date_id'], ret['meta_time']) for ret in to_analyze]
    else:
      args = [(ret['name'], ret['file_date'], ret['file_date_id'], 0) for ret in to_analyze]
    print(json.dumps(args))

    print(prefix)
    job_file = os.path.join(job_directory,  prefix + '.job')

    with open(job_file, 'w') as fh:
      fh.writelines("#!/bin/bash\n")
      fh.writelines("#SBATCH --job-name=%s\n" % prefix)
      fh.writelines("#SBATCH --output=%s.out\n" % os.path.join(job_directory, 'out', '%x.%A_%a.%N.%j'))
      fh.writelines("#SBATCH --error=%s.err\n" % os.path.join(job_directory, 'err', '%x.%A_%a.%N.%j'))
      fh.writelines("#SBATCH --time=0-8\n")  # hours
      fh.writelines("#SBATCH --mem=48G\n")  # memory for each task
      fh.writelines("#SBATCH -p gpu_requeue\n")  # partition (queue)
      fh.writelines("#SBATCH --gres=gpu:1\n")
      fh.writelines("#SBATCH -N 1\n")  # nodes
      fh.writelines("#SBATCH -c 1\n")  # cpus
      fh.writelines("#SBATCH --mail-type=END,FAIL,TIME_LIMIT\n")
      fh.writelines("#SBATCH --mail-user=alowet@g.harvard.edu\n")
      fh.writelines("#SBATCH --export=ALL\n")

      fh.writelines("module load python/3.10.9-fasrc01\n")
      fh.writelines("module load cuda/11.8.0-fasrc01\n")
      fh.writelines("module load cudnn/8.8.0.121_cuda12-fasrc01\n")
      fh.writelines("source activate tf\n")
      # escape string with '' so it gets treated as a single argument that can then be read by json
      fh.writelines("python3 -u regress_chunk.py '{}' -t {} -r {} -l {} -g {} -f {} -i {}\n".format(
        json.dumps(args), table, refit, lr, reg, se_frac, l1_ratio))

    os.system("sbatch %s" % job_file)

#    break
#  break

#  else:
#    print('Already analyzed in full. Skipping chunk ' + str(i_chunk))
#  break
#  if i_chunk == 5:
#    break

print(job_count)
print(job_count * n_sessions_per_chunk)
