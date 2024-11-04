import os
from loop_suite2p import loop_suite2p

import sys
sys.path.append('../../utils')
from db import get_db_info

RERUN = 0
paths = get_db_info()

session_folders = []
for root, dirs, files in os.walk(paths['imaging_root']):
    for file in files:
        if file.endswith('.tif'):
            path = os.path.dirname(os.path.join(root, file))
            # don't add if suite2p folder already exists, meaning video has been analyzed already
            s2p = os.path.join(path, 'suite2p')
            if not os.path.isdir(s2p) or len(os.listdir(s2p)) == 0 or RERUN:
                session_folders.append(path)

unique_folders = list(set(session_folders))
print(unique_folders)

# Run a separate job for each recording session
for sess in unique_folders:

    print('Running Suite2P on ' + sess)

    fileparts = os.path.normpath(sess).split('/')
    subj_name = fileparts[-2]
    date = fileparts[-1]

    loop_suite2p(os.path.join(paths['imaging_root'], subj_name, date))

