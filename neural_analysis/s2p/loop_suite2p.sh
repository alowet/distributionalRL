#!/usr/bin/env python

#SBATCH -J Suite2PLoop
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 2000
#SBATCH -t 10 # minutes
#SBATCH -o job-scripts/out/Job.%x.%A_%a.%N.%j.out # STDOUT
#SBATCH -e job-scripts/err/Job.%x.%A_%a.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL,TIME_LIMIT # notifications
#SBATCH --mail-user=alowet@g.harvard.edu # send-to address

"""
Usage: sbatch loop_suite2p.sh

Spawn a new, multithreaded job of suite2p for each session to analyze.
Must edit data_root and ssd_root to your directories.

Written by Adam S. Lowet, Nov. 20, 2019
"""

import os, sys
sys.path.append('../../utils')
from paths import check_dir

data_root = '/n/holystore01/LABS/uchida_users/Users/alowet/2P-microscope/'
ssd_root = os.path.join(os.environ['SCRATCH'], 'uchida_lab', 'alowet', 'neural')
#ssd_root = '/n/holylfs04-ssd2/LABS/uchida_users/Users/alowet/imaging'
home_root = '/n/home06/alowet/dist-rl/data'

RERUN = 0  # do we want to rerun analyses that have already been performed? (presumably with different parameters)
# do we want to delete EVERYTHING from the SSD path, including the registered binary files?
# They will be transferred to the save_path regardless, and suite2p will 
DELETE_SSD = 1

session_folders = []
for root, dirs, files in os.walk(data_root):
	for file in files:
		if file.endswith('.tif'):
			path = os.path.dirname(os.path.join(root, file))
			# don't add if suite2p folder already exists, meaning video has been analyzed already
			s2p = os.path.join(path, 'suite2p')
			if not os.path.isdir(s2p) or len(os.listdir(s2p)) == 0 or RERUN:
				session_folders.append(path)

unique_folders = list(set(session_folders))
print(unique_folders)
job_directory = "./job-scripts"

# Make top level directories
# mkdir_p(job_directory)
check_dir(os.path.join(job_directory, 'out'))
check_dir(os.path.join(job_directory, 'err'))

# Run a separate job for each recording session
for sess in unique_folders:

	print('Running Suite2P on ' + sess)
	
	fileparts = os.path.normpath(sess).split('/')
	subj_name = fileparts[-2]
	date = fileparts[-1]

	ssd_path = os.path.join(ssd_root, subj_name, date)
	check_dir(ssd_path)
	home_path = os.path.join(home_root, subj_name, date, 'suite2p')
	check_dir(home_path)

	sess_name = '_'.join(fileparts[-2:])
	job_file = os.path.join(job_directory, sess_name + '.job')

	with open(job_file, 'w') as fh:
		fh.writelines("#!/bin/bash\n")
		fh.writelines("#SBATCH --job-name=%s\n" % sess_name)
		fh.writelines("#SBATCH --output=%s.out\n" % os.path.join(job_directory,'out','%x.%A_%a.%N.%j'))
		fh.writelines("#SBATCH --error=%s.err\n" % os.path.join(job_directory,'err','%x.%A_%a.%N.%j'))
		fh.writelines("#SBATCH --time=0-04:00\n")
		fh.writelines("#SBATCH --mem=32G\n") #memory for each task
		fh.writelines("#SBATCH -p shared\n") # partition (queue)
		fh.writelines("#SBATCH -N 1\n") # nodes
		fh.writelines("#SBATCH -c 8\n") # cpus
		fh.writelines("#SBATCH --mail-type=END,FAIL,TIME_LIMIT\n")
		fh.writelines("#SBATCH --mail-user=alowet@g.harvard.edu\n")
		fh.writelines("#SBATCH --export=ALL\n")

		# fh.writelines("module load Anaconda3/5.0.1-fasrc02\n")
		fh.writelines("module load python/3.10.9-fasrc01\n")
		fh.writelines("source activate suite2p\n")
    # fh.writelines("export OMP_NUM_THREADS=1\n")
    # fh.writelines("export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK\n")
		# fh.writelines("export OMP_STACKSIZE='16G'\n")
		# fh.writelines('srun -n $SLURM_CPUS_PER_TASK fpsync -n $SLURM_CPUS_PER_TASK -o "--ax" -O "-b" %s %s 0\n' %(sess, os.path.join(ssd_root, subj_name)))
		fh.writelines("rsync -avx --progress %s %s\n" %(sess, os.path.join(ssd_root, subj_name)))
		fh.writelines("python3 -u loop_suite2p.py %s %s %s\n" %(ssd_path, ssd_path, sess))
		# copy EVERYTHING, including the registered binaries to the save_path
		fh.writelines("rsync -avx --progress %s %s\n" %(os.path.join(ssd_path, 'suite2p'), sess))
		fh.writelines("rsync -avx --progress --include=*/ --include=*.npy --exclude=* %s %s\n" %(os.path.join(ssd_path, 'suite2p', 'plane0'), home_path))
		# remove files from ssd_path 
		if DELETE_SSD:
			fh.writelines("rm -r %s\n" %(ssd_path))  # remove everything
		else:
			fh.writelines("rm %s\n" %(os.path.join(ssd_path, '*.tif')))  # this way the registered binaries should stay, if I want them to
		fh.writelines("echo suite2p complete on file %s\n" % sess_name)
	os.system("sbatch %s" %job_file)

