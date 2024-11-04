import os
import glob
import multiprocessing as mp
from analyze_pupil import analyze_pupil
import sys
sys.path.append('../db')
from dbutils import get_db_info, on_cluster

RERUN = 0
SAVE_FIGS = 1
paths = get_db_info()

if on_cluster():
	try:
		pool = mp.Pool(int(os.environ['SLURM_CPUS_PER_TASK']))  # only works if run as sbatch loop_behavior.sh
	except KeyError:
		pool = mp.Pool(16)  # my interactive session default
else:
	pool = mp.Pool(7)  # leave one physical CPU for my use

# loop through the pupil root
pupil_paths_to_analyze = []
for mouse_name in os.listdir(paths['pupil_root']):
	subj_folder = os.path.join(paths['pupil_root'], mouse_name)
	if os.path.isdir(subj_folder):
		for file_date_id in os.listdir(subj_folder):
			sess_folder = os.path.join(subj_folder, file_date_id)
			if os.path.isdir(sess_folder):
				csv_name = glob.glob(os.path.join(sess_folder, '*.csv'))
				if len(csv_name) == 1 and (RERUN or not glob.glob(os.path.join(sess_folder, '*.npz'))):
					pupil_paths_to_analyze.append(sess_folder)
				elif len(csv_name) > 1:
					raise Exception('Too many CSVs in ' + sess_folder)

pool.map(analyze_pupil, pupil_paths_to_analyze)
pool.close()
pool.join()

print('Pupil loop complete.')