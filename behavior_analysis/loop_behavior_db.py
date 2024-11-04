"""
Loop through all .mat files contained within a given root directory.
If file meets criteria, run plotSingleSessionDistributionalRL6Odours.py on it.

Written by Adam S. Lowet, Nov. 2019
"""
import os
from plotSingleSession import plotSingleSession
import subprocess
import sys
import glob
import multiprocessing as mp
sys.path.append('../utils')
from db import get_db_info, create_connection, on_cluster

paths = get_db_info()

SAVE_FIGS = 1
RERUN = 0

conn = create_connection(paths['db'])
cur = conn.cursor()
if RERUN:
    cur.execute('SELECT name, protocol, exp_date, has_opto FROM session WHERE protocol="SameRewVar"')
else:
    cur.execute('SELECT name, protocol, exp_date, has_opto FROM session WHERE date_analyzed IS NULL')
ret = cur.fetchall()
conn.close()

try:
    pool = mp.Pool(os.environ['SLURM_NCPUS_PER_TASK'])  # only works if run as sbatch loop_behavior.sh
except KeyError:
    pool = mp.Pool(16)

print(ret)
for tup in ret:
    name = tup[0]; protocol = tup[1]; exp_date = str(tup[2]); has_opto = tup[3]
    # for data_root in paths['all_behavior_roots']:
    try_path = os.path.join(paths['behavior_root'], name, protocol, 'Session Data', name + '_' + protocol + '_' + exp_date + '*')
    matches = glob.glob(try_path)
    if len(matches) > 0:
        # print(matches)
        pool.apply_async(plotSingleSession, args=(name, protocol, exp_date, SAVE_FIGS, has_opto))
        # pool.apply_async(plotSingleSession, args=(paths['behavior_root'], name, protocol, exp_date, paths['behavior_fig_roots'], SAVE_FIGS))
        # plotSingleSession(name, protocol, exp_date, SAVE_FIGS, has_opto)
    else:
        print("Couldn't find any matches. Are you sure the data transferred successfully?")
pool.close()
pool.join()

# if on_cluster():
#     # transfer plots/stats from scratch to long-term storage
#     # subprocess.call(['rsync', '-avx', '--progress', plot_roots, transfer_root])
#     # clean up scratch with an rm
#     scratch_data_folder = os.path.join(os.environ['SCRATCH'], 'uchida_lab', 'alowet', 'behavior')
#     if os.path.isdir(scratch_data_folder):
#         subprocess.call(['rm', '-r', scratch_data_folder])

