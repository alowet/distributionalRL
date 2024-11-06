import os
import sys
import subprocess
sys.path.append('../utils')
from db import get_db_info, select_db

paths = get_db_info()
rets = select_db(paths['db'], 'ephys', '*', 'significance=1', (), unique=False)
for ret in rets:
    ks_path = os.path.join('/mnt/nas/ephys', ret['name'], ret['file_date_id'], 'ks_matlab')
    dest_path = os.path.join(paths['remote_ephys_root'],  ret['name'], ret['file_date_id'], 'ks_matlab')
    subprocess.run(['rsync', '-avx', '--progress', os.path.join(ks_path, 'templates.npy'),
                    'alowet@login.rc.fas.harvard.edu:' + dest_path])
    subprocess.run(['rsync', '-avx', '--progress', os.path.join(ks_path, 'spike_templates.npy'),
                    'alowet@login.rc.fas.harvard.edu:' + dest_path])