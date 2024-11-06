"""
Loop through all preprocessed and curated ephys sessions, classifying units and stashing in db

Written by Adam S. Lowet, Dec 2020
"""
from classify_units import classify_units
import sys
sys.path.append('../utils')
from db import get_db_info, select_db

RERUN = 0
paths = get_db_info()
rets = select_db(paths['db'], 'ephys', '*', 'curated=1 AND significance=1 AND name>="AL27"', (), unique=False)
for ret in rets:
    classify_units(ret['name'], ret['file_date_id'], RERUN)
print('Classification loop complete.')