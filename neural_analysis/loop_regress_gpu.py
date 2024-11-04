"""
Usage: sbatch loop_regress_gpu.py

Spawn a new GPU job for each session to fit with regress_session.py

Written by Adam S. Lowet, Feb 14, 2024
"""

import os, sys

sys.path.append('../utils')
from paths import check_dir
from db import select_db, get_db_info, execute_sql
import random
import json

RERUN = 0  # do we want to rerun analyses that have already been performed? (presumably with different parameters)
n_sessions_per_chunk = 6

table = 'ephys'
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

# Run a separate job for each neuron that has not been analyzed already
random.Random(1).shuffle(rets)
chunks = [rets[i:i+n_sessions_per_chunk] for i in range(0, len(rets), n_sessions_per_chunk)]
print(len(chunks))
# print(chunks)

for chunk in chunks:
  to_analyze = []
  if RERUN: to_analyze = chunk
  else:
    for ret in chunk:
      in_table = select_db('my', 'glm_setup', '*', 'mid=? AND sid=? AND rid=?', (ret['mid'], ret['sid'], ret['rid']), unique=False)
      if len(in_table) < 4: to_analyze.append(ret)

  if len(to_analyze) > 0:
      prefix = '_'.join(['_'.join([ret['name'], ret['file_date_id']]) for ret in to_analyze])
      args = [(ret['name'], ret['file_date'], ret['file_date_id']) for ret in to_analyze]
      print(json.dumps(args))

      print(prefix)
      job_file = os.path.join(job_directory,  prefix + '.job')

      with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s\n" % prefix)
        fh.writelines("#SBATCH --output=%s.out\n" % os.path.join(job_directory, 'out', '%x.%A_%a.%N.%j'))
        fh.writelines("#SBATCH --error=%s.err\n" % os.path.join(job_directory, 'err', '%x.%A_%a.%N.%j'))
        fh.writelines("#SBATCH --time=0-4\n")  # hours
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
        fh.writelines("python3 -u regress_chunk.py '%s'\n" % (json.dumps(args)))
      #  os.system("sbatch %s" % job_file)
  break

