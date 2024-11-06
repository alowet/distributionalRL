"""
Loop through all preprocessed and curated ephys sessions, plotting neurons

Written by Adam S. Lowet, Dec 2020
"""
import os
import glob
import multiprocessing as mp
from plot_spikes import plot_spikes
from datetime import datetime
from itertools import repeat
import sys
sys.path.append('../utils')
from db import get_db_info, select_db

RERUN = 0
PLOT = 1
paths = get_db_info()

if RERUN:
    rets = select_db(paths['db'], 'ephys', '*', 'curated=1 AND name<"AL87" AND date_processed<20231006', (), unique=False)  #datetime.today().strftime('%Y%m%d')
    # rets = select_db(paths['db'], 'ephys', '*', 'date_curated>20210609 AND curated=1', (), unique=False)
else:
    rets = select_db(paths['db'], 'ephys', '*', 'date_processed IS NULL AND curated=1', (), unique=False)
# print(rets)
proc_paths = [os.path.dirname(ret['processed_data_path']) for ret in rets]
# print(proc_paths)

# pool = mp.Pool(mp.cpu_count())
pool = mp.Pool(int(os.environ['SLURM_CPUS_PER_TASK']))
out = pool.starmap(plot_spikes, zip(proc_paths, repeat(PLOT), repeat(RERUN)), chunksize=1)
pool.close()
pool.join()
print(out)

# for path in proc_paths:
#     plot_spikes(path, plot=PLOT, rerun=RERUN)

print('Spike loop complete.')