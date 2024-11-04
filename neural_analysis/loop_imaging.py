"""
Loop through all preprocessed and curated imaging sessions, plotting neurons

Written by Adam S. Lowet, Feb 2020
"""
import os
import multiprocessing as mp
from plotTrialAvgsByNeuron import plotTrialAvgsByNeuron
from analyze_pupil import analyze_pupil
import sys
import tqdm

sys.path.append('../utils')
from db import on_cluster, get_db_info, create_connection

ON_CLUSTER = on_cluster()
paths = get_db_info()

RERUN = 1

# encapsulate these two processing steps into a single function for multiprocessing pool
def group_analysis(pupil_dir, imaging_dir):
	analyze_pupil(pupil_dir)
	plotTrialAvgsByNeuron(imaging_dir)


if ON_CLUSTER:
	try:
		pool = mp.Pool(int(os.environ['SLURM_CPUS_PER_TASK']))  # only works if run as sbatch loop_imaging.sh
	except KeyError:
		pool = mp.Pool(16)
else:
	pool = mp.Pool(7)  # leave one physical CPU for my use

conn = create_connection(paths['db'])
cur = conn.cursor()

# curated must be equal to 1. -1 is my convention to indicate that it was looked at, and there were no good cells
# 0 is default; means it hasn't yet been looked at, 2 means it's bad, but still want to play with the data
if RERUN:
	cur.execute('SELECT name, file_date_id, continuous, file_date, mid FROM imaging WHERE curated=1 AND name="AL06" AND file_date_id!="20200629"')  #name>="AL60" AND name<"AL70"  # AND file_date>20220801 AND continuous=0 AND name="AL60"')
else:
	cur.execute('SELECT name, file_date_id, continuous, file_date, mid FROM imaging WHERE date_processed IS NULL AND curated=1')
ret = cur.fetchall()
conn.close()

imaging_dirs = [os.path.join(paths['imaging_root'], tup[0], str(tup[1])) for tup in ret]
print(imaging_dirs)
results = list(tqdm.tqdm(pool.imap(plotTrialAvgsByNeuron, imaging_dirs, chunksize=1), total=len(ret)))

pool.close()
pool.join()

print('Imaging loop complete.')
