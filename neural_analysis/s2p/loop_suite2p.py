import numpy as np
from suite2p.run_s2p import run_s2p
import sys
import os
from s2p_db_utils import get_kludge, load_ops
import scipy.io as sio
from datetime import datetime
import pickle
import json
from ScanImageTiffReader import ScanImageTiffReader
import glob
import subprocess
from threadpoolctl import threadpool_limits

sys.path.append('../../utils')
from db import *
from paths import check_dir

"""
Meant to be called from loop_suite2p.sh
Sets options and runs suite2p for a given data_path, fast_disk, and save_path, passed from loop_suite2p.sh
"""


def loop_suite2p(data_path, fast_disk=None, save_path=None, sweep=None, ctr=None):

    print(save_path)
    if fast_disk is None:
        fast_disk = data_path
    if save_path is None:
        save_path = data_path

    # Database location and settings:
    paths = get_db_info()
    insert_data = {}
    # subject's name is the second-to-last directory of data_path
    insert_data['name'] = os.path.split(os.path.split(data_path)[0])[1]
    # file date is the last directory of data path
    insert_data['file_date_id'] = os.path.split(data_path)[1]
    insert_data['file_date'] = os.path.split(data_path)[1][:8]

    # Get code from database
    ret = select_db(paths['db'], 'mouse', 'mid, code', 'name=?', (insert_data['name'],))
    insert_data['mid'] = ret['mid']
    code = int(ret['code'])

    if code is None:
        raise Exception('No code associated with this mouse in the database. Check Google Spreadsheet and try again.')

    # load the options associated with that code
    intermediate_save_path = data_path
    if ctr is not None:
        intermediate_save_path = os.path.join(intermediate_save_path, ctr)
        check_dir(intermediate_save_path)

    # get tiff metadata
    base_tiff_regex = '_'.join([insert_data['name'], insert_data['file_date'], '00?', '001.tif'])
    base_tiff = glob.glob(os.path.join(data_path, base_tiff_regex))[0]
    reader = ScanImageTiffReader(base_tiff)
    meta = dict(x.split(' = ') for x in reader.description(0).split('\n', maxsplit=150))
    reader.close()

    code = get_kludge(insert_data, code)
    ops = load_ops(code, intermediate_save_path, insert_data, meta)

    if sweep is not None:
        # nice bit of syntax to replace ops if key exists in sweep
        ops = {**ops, **sweep}

    # provide an h5 path in 'h5py' or a tiff path in 'data_path'
    # db overwrites any ops (allows for experiment specific settings)
    db = {
        'h5py': [],  # a single h5 file path
        'h5py_key': 'data',
        'look_one_level_down': False,  # whether to look in ALL subfolders when searching for tiffs
        'data_path': [data_path],  # a list of folders with tiffs
        # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)

        'subfolders': [],  # choose subfolders of 'data_path' to look in (optional)
        'fast_disk': fast_disk,  # string which specifies where the binary file will be stored (should be an SSD)
    }

    # run one experiment
    try:
        # with threadpool_limits(limits="sequential_blas_under_openmp"):
        opsEnd = run_s2p(ops=ops, db=db)
    except:
        # sometimes tiff reading fails with ScanImageTiffReader, and if it fails after the first tiff,
        # suite2p will just error out. Catch this error and use scikit tiff reader instead
        ops['force_sktiff'] = True
        print('Using scikit-image to read TIFFs')
        # with threadpool_limits(limits="sequential_blas_under_openmp"):
        opsEnd = run_s2p(ops=ops, db=db)

    if sweep is not None:
        print('Saving to ' + intermediate_save_path)
        pickle.dump(sweep, open(os.path.join(intermediate_save_path, 'params.p'), 'wb'))

    else:
        # collect all database items
        metadata = sio.loadmat(os.path.join(data_path, 'metadata.mat'))
        meta_date = metadata['meta_date'][0]
        insert_data['meta_date'] = int(datetime.strptime(meta_date, '%m/%d/%Y').strftime('%Y%m%d'))
        meta_time = metadata['meta_time'][0]
        insert_data['meta_time'] = int(datetime.strptime(meta_time, '%I:%M %p').strftime('%H%M'))
        insert_data['raw_data_path'] = save_path.replace(paths['imaging_root'], paths['remote_imaging_root'])
        insert_data['processed_data_path'] = save_path.replace(paths['imaging_root'], paths['remote_imaging_root'])
        insert_data['date_suite2p'] = datetime.today().strftime('%Y%m%d')
        insert_data['suite2p_ops'] = json.dumps(ops, cls=NumpyEncoder)  # from s2p_db_utils; handles np arrays and makes them serializable
        insert_data['curated'] = 0
        insert_data['fs'] = ops['fs']
        insert_data['nchannels'] = ops['nchannels']
        insert_data['functional_chan'] = ops['functional_chan']

        insert_data['beam_power'] = int(meta['scanimage.SI4.beamPowers'])
        insert_data['chans_saved'] = meta['scanimage.SI4.channelsSave']
        insert_data['zoom'] = float(meta['scanimage.SI4.scanZoomFactor'])

        if insert_data['meta_date'] > 20200219 and insert_data['meta_date'] < 20220528:  # when I was using continuous acquisition
            insert_data['continuous'] = 1

        # stash results in metadatabase
        insert_into_db(paths['db'], 'imaging', tuple(insert_data.keys()), tuple(insert_data.values()))
        print('Inserted data into imaging table for file ' + save_path)

    if not on_cluster():
        imaging_path = os.path.join(paths['imaging_root'], insert_data['name'], insert_data['file_date_id'])
        subprocess.call(['rsync', '-avx', '--progress', imaging_path, 'alowet@login.rc.fas.harvard.edu:' +
                         os.path.join(paths['remote_imaging_root'], insert_data['name'])])
        home_path = os.path.join('/mnt/clusterhome/dist-rl/data', insert_data['name'], insert_data['file_date_id'], 'suite2p')
        check_dir(home_path)
        subprocess.call(['rsync', '-avx', '--progress', '--include=*/', '--include=*.npy', '--exclude=*',
                         os.path.join(imaging_path, 'suite2p', 'plane0'), home_path])

if __name__ == '__main__':

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        loop_suite2p(sys.argv[1])
    elif len(sys.argv) == 4 and os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2]) and os.path.isdir(sys.argv[3]):
        loop_suite2p(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 6:
        loop_suite2p(sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), sys.argv[5])
    else:
        raise ('Invalid file specified. Usage: python loop_suite2p.py data_path fast_disk save_path')
