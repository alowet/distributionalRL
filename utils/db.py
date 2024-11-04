import socket
import sqlite3
import mysql.connector
from sqlite3 import Error
import numpy as np
import io
import json
import os
import glob
import time
from datetime import datetime
import pickle
import subprocess
from paths import copy_behavior, parse_data_path, get_names_tosave, raise_print
from matio import loadmat

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def on_cluster():
	if 'rc.fas.harvard.edu' in socket.gethostname():
		ON_CLUSTER = True
	else:
		ON_CLUSTER = False
	return ON_CLUSTER


def get_db_info():
    paths = {'db': '../data/session_log.sqlite',
                'config': '../data/session_log_config.json',
                'home_root': '../data',
                'behavior_root': '../data/behavior',
                'imaging_root': '../data/imaging',
                'pupil_root': '../data/data/pupil',
                'behavior_fig_roots': ['../behavior-plots',
                                       '../data/behavior-plots'],
                'neural_fig_roots': ['../data/neural-plots',
                                    '../data/neural-plots'],
                'ephys_root': '../data/ephys',
                'facemap_root': '../data/camera',
                'brainglobe_dir': '../.brainglobe'
            }

    for key in ['behavior', 'imaging', 'ephys', 'facemap', 'pupil', 'behavior_fig', 'neural_fig']:
        paths['_'.join(['remote', key, 'root'])] = paths['_'.join([key, 'root'])]

	return paths


def create_connection(db_file, il=True, uri=False):
	""" create a database connection to the SQLite database
		specified by db_file
	:param db_file: database file
	:param il: isolation_level, used to allow write-ahead logging (or try to)
	:param uri: read only mode
	:return: Connection object or None
	"""
	conn = None
	try:
		if db_file == 'my':
			conn = mysql.connector.connect(
				host='rcdb-user.rc.fas.harvard.edu',
				user='alowet',
				password='d72f4T6Xwz!1te',
				database="alowet"
				)
		# 30 second timeout to handle concurrency better
		elif il:
			conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, timeout=30, uri=uri)
		else:
			conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, timeout=30, isolation_level=il, uri=uri)
		return conn
	except Error as e:
		raise Exception(e)
	except mysql.connector.Error as e:
		raise Exception(e)

	return conn


def execute_sql(sql, db_file, keys=True):
	conn = create_connection(db_file)
	if keys and db_file == 'my':
		cur = conn.cursor(dictionary=True)
	else:
		if keys:
			conn.row_factory = sqlite3.Row
		cur = conn.cursor()
	cur.execute(sql)
	ret = cur.fetchall()
	conn.close()
	return ret


def execute_many(sql, db_file, seq_of_params):
	conn = create_connection(db_file)
	cur = conn.cursor()
	cur.executemany(sql, seq_of_params)
	conn.commit()
	cur.close()
	conn.close()


def update_db(sql, db_file):
	conn = create_connection(db_file)
	cur = conn.cursor()
	cur.execute(sql)
	conn.commit()
	cur.close()
	conn.close()


def select_db(db, table, select_str, where_str, where_vals, keys=True, unique=True):
	"""
	:param db: path to database
	:param table: name of table in database
	:param select_str: comma-separated string of columns to select, or '*'
	:param where_str: AND-separated string of columns to select upon, e.g. 'name=? AND file_date_id=?'
	:param where_vals: tuple of values to use in selection, with column-names given in where_str
	:return:
	"""
	conn = create_connection(db)
	if db == 'my':
		if keys:
			cur = conn.cursor(dictionary=True)
		else:
			cur = conn.cursor()
		where_str = where_str.replace('?', '%s')

	else:
		if keys:
			conn.row_factory = sqlite3.Row
		cur = conn.cursor()
	
	cur.execute('SELECT ' + select_str + ' FROM ' + table + ' WHERE ' + where_str, where_vals)

	ret = cur.fetchall()
	conn.close()

	if unique:
		if len(ret) != 1:
			raise Exception('Multiple entries (or zero entries) found with where string={}, vals={}. Please manually select entry'.format(where_str, where_vals))
		return ret[0]
	return ret


def get_column_names(dbconfig, fields):
	# get column names from database
	with open(os.path.join(dbconfig)) as json_file:
		db_info = json.load(json_file)
	return list([x[0] for x in db_info[fields]])  # keys in the db


def insert_into_db(db, table, col_names, insert_vals, wal=False, many=False):
	if wal:
		conn = create_connection(db, il=None)
		conn.execute('pragma journal_mode=wal')
	else:
		conn = create_connection(db)

	if db == 'my':
		filler = '%s'
		stmt = 'REPLACE INTO '

	else:
		filler = '?'
		stmt = 'INSERT OR REPLACE INTO '
	
	cur = conn.cursor()
	sql = stmt + table + ' (' + ', '.join(col_names) + ') VALUES (' + ', '.join(
		[filler] * len(col_names)) + ')'
	# flag = True
	# while flag:
		# try:
	if many:
		cur.executemany(sql, insert_vals)
	else:
		cur.execute(sql, insert_vals)
			# flag = False
		# except sqlite3.OperationalError:  # this error occurs on commit, not on execute!
		# 	# if multiple writes at once because of multiprocessing, wait a random amount of time up to 1 minute and try again
		# 	time.sleep(np.random.default_rng().random()*60)
	# except mysql.connr.errors.IntegrityError:
	conn.commit()
	conn.close()


def adapt_array(arr):
	"""
	http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
	"""
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(out.read())


def convert_array(text):
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out)


def add_to_db_dict(db_dict, foldernames_tosave):

	db_dict['date_processed'] = datetime.today().strftime('%Y%m%d')
	db_dict['figure_path'] = foldernames_tosave[0]

	if not on_cluster():
		# get the git hash of the latest commit, to keep track of version of the script that ran
		db_dict['git_hash'] = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")

	return db_dict


def analyze_neural(data_dir, table):

	# Conglomeration of functions that are common to 2p imaging analysis and NPX analysis
	
	paths = get_db_info()
	mouse_name, file_date, file_date_id = parse_data_path(data_dir)
	print(mouse_name, file_date, file_date_id)

	# make a folder to save our figures if it doesn't exist
	foldernames_tosave, filenames_tosave = get_names_tosave(mouse_name, file_date, file_date_id, paths['neural_fig_roots'])

	# Get column names and data from database
	db_entry = select_db(paths['db'], table, '*', 'name=? AND file_date_id=?', (mouse_name, file_date_id))
	db_dict = {k: db_entry[k] for k in db_entry.keys()}  # convert it to a dict so that it is mutable

	db_dict = add_to_db_dict(db_dict, foldernames_tosave)

	# load behavior data. right now, assumes one recording sess per day per protocol, but could easily revert by setting
	# unique=False in the select_db call
	sessions = select_db(paths['db'], 'session', 'protocol', 'name=? AND exp_date=? AND has_' + table + '=1', (mouse_name, file_date), unique=False)
	assert(np.all([sess['protocol'] == sessions[0]['protocol'] for sess in sessions]))
	protocol = sessions[0]['protocol']
	behavior_path = os.path.join(paths['behavior_root'], mouse_name, protocol, 'Session Data')
	
	# format is different depending on the table
	if table == 'imaging':
		meta_time = datetime.strptime(str(db_dict['meta_time']), '%H%M')
	else:
		meta_time = datetime.strptime(str(db_dict['meta_time']), '%H%M%S')

	behavior_filepath, session_data = get_session(behavior_path, file_date, meta_time, table)
	db_dict['behavior_path'] = behavior_filepath

	# get session_cmt field from session db and insert it into ephys db
	remote_behavior_filepath = behavior_filepath.replace(paths['behavior_root'], paths['remote_behavior_root'])
	print(remote_behavior_filepath)
	behavior_entry = select_db(paths['db'], 'session', 'session_cmt', 'raw_data_path=?', (remote_behavior_filepath,))
	db_dict['notes'] = behavior_entry['session_cmt']

	try:
		# Extract the number that precedes 'nm'
		db_dict['wavelength'] = int(db_dict['notes'].split('nm')[0].strip().split(' ')[-1])
	except (ValueError, TypeError, AttributeError):  # not numeric, or I forgot to include this in the notes
		pass

	# load pupil data, if it exists
	pupil = {}
	if db_dict['pupil_path']:
		local_pupil_path = db_dict['pupil_path'].replace(paths['remote_pupil_root'], paths['pupil_root'])
		pupil['fname'] = glob.glob(local_pupil_path + '*.p')[0]
		pupil['dat'] = pickle.load(open(pupil['fname'], 'rb'))

	behavior_filename = copy_behavior(behavior_filepath, paths['behavior_fig_roots'][0], foldernames_tosave, mouse_name, file_date)

	# get behavior data
	behavior = {}
	behavior['dat'] = pickle.load(open(os.path.join(paths['behavior_fig_roots'][0], db_dict['name'],
	                                              str(db_dict['file_date']), behavior_filename + '.p'), 'rb'))

	names_tosave = {'foldernames': foldernames_tosave, 'filenames': filenames_tosave}

	return paths, db_dict, session_data, protocol, names_tosave, pupil, behavior


def get_session(beh_data_folder, day, time, table=None):
	print(beh_data_folder, day, time, table)
	session = glob.glob(beh_data_folder + '/*' + day + '*.mat')
	datafile_names = []
	sessions = []
	file_times = []
	print(session)
	for df in session:
		converted_data = loadmat(os.path.join(beh_data_folder, df))
		session_data = converted_data['SessionData']
		# print(session_data)
		print(session_data['has_' + table])
		# kludge from where the field in the table was called imaging/image instead of has_imaging
		for old_key in ['imaging', 'image']:
			if old_key in session_data:
				session_data['has_imaging'] = session_data[old_key]
		if table is None or session_data['has_' + table] == 1:
			print('inside')
			# kludge for old days when Bpod was freezind and sometimes returning an empty list
			if 'quality' in session_data and isinstance(session_data['quality'], int):
				datafile_names.append(df)
				sessions.append(session_data)
				if 'exp_time' in session_data:
					file_times.append(session_data['exp_time'])
				else:  # kludge for when this used to be called 'file_time'
					file_times.append(session_data['file_time'])
	if len(datafile_names) == 1:
		return datafile_names[0], sessions[0]
	elif len(datafile_names) > 1:
		if table == 'imaging':
			# figure out which (behavior) file_time is closest to (imaging) meta_time, and use that behavior file
			date = datetime.strptime(day, '%Y%m%d').date()
			dt = datetime.combine(date, time.time())
			which_session = np.argmin(np.abs([dt - datetime.combine(date, datetime.strptime(file_time, '%H%M%S').time()) for file_time in file_times]))
		else: # just use largest file in the event the above session selects wrong one (e.g. AL28/20210325)
			which_session = np.argmax([os.path.getsize(df) for df in datafile_names])
		print(which_session)
		return datafile_names[which_session], sessions[which_session]
	else:
		raise_print("Couldn't find " + table + " session on this day.")
