import os
import glob
import pickle
import subprocess

def raise_print(message):
	print(message)  # so it gets logged in the stdout file from SLURM
	raise Exception(message)


def check_dir(dest_dir):
	if not os.path.isdir(dest_dir):
		try:
			os.makedirs(dest_dir)
		except FileExistsError:
			# because multiple processes are running, might now exist
			pass


def splitall(path):
	allparts = []
	while 1:
		parts = os.path.split(path)
		if parts[0] == path:  # sentinel for absolute paths
			allparts.insert(0, parts[0])
			break
		elif parts[1] == path:  # sentinel for relative paths
			allparts.insert(0, parts[1])
			break
		else:
			path = parts[0]
			allparts.insert(0, parts[1])
	return allparts


def parse_data_path(data_path):
	"""
	Extract mouse_name, file_date_id, and file_date from a path that is formatted as
	/path/to/file/mouse_name/file_date_id
	"""
	mouse_name = os.path.split(os.path.split(data_path)[0])[1]
	# file date is the last directory of data path
	file_date_id = os.path.split(data_path)[1]
	file_date = file_date_id[:8]
	return mouse_name, file_date, file_date_id


def save_pickle(fig_paths, to_save, suffix, data_path=None):
	# save pickle file only to 0th folder specified
	if not data_path:
		data_path = fig_paths[0]
	with open(data_path + '_' + suffix + '.p', 'wb') as f:
		pickle.dump(to_save, f)

def get_names_tosave(mouse_name, file_date, file_date_id, fig_root):
	# make a folder to save our figures if it doesn't exist
	foldernames_tosave = [os.path.join(x, mouse_name, file_date_id) for x in fig_root]
	[check_dir(x) for x in foldernames_tosave]
	filenames_tosave = [os.path.join(x, mouse_name + '_' + file_date) for x in foldernames_tosave]
	return foldernames_tosave, filenames_tosave

def copy_behavior(behavior_filepath, behavior_fig_root, foldernames_tosave, mouse_name, file_date):
	# copy behavior figs/pickle files into processed_data_path directory
	behavior_filename = os.path.basename(behavior_filepath).replace('.mat', '')
	behavior_files = glob.glob(
		os.path.join(behavior_fig_root, mouse_name, file_date, behavior_filename + '*'))
	for file in behavior_files:
		for i_folder, foldername_tosave in enumerate(foldernames_tosave):
			if i_folder == 0:
				subprocess.call(['rsync', '-avx', '--progress', file, foldername_tosave])
			# don't copy over anything besides png files to home directory
			elif file[-4:] == '.png':
				subprocess.call(['rsync', '-avx', '--progress', file, foldername_tosave])
	return behavior_filename