import os
import subprocess
import sys
sys.path.append('../utils')
from db import get_db_info, select_db
from paths import check_dir

paths = get_db_info()
rets = select_db(paths['db'], 'ephys', '*', 'registered=1', (), unique=False)
local_ephys_root = '/mnt/nas2/ephys'
for ret in rets:
	source_file = os.path.join(local_ephys_root, ret['name'], str(ret['file_date']), 'alf', 'channel_locations.json')
	dest_dir = os.path.join(paths['ephys_root'], ret['name'], str(ret['file_date']))
	check_dir(dest_dir)
	# subprocess.call(['rsync', '-avx', '--progress', pdfname_tosave, names_tosave['foldernames'][1]])
	subprocess.run(['rsync', '-avx', '--progress', source_file, dest_dir])
