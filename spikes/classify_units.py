import numpy as np
from scipy.stats import mode
import pandas as pd
import os
import time
import sys
import json
from phylib.stats import correlograms
sys.path.append('../utils')
from db import get_db_info, select_db, on_cluster, insert_into_db, execute_sql, execute_many
import pickle

def classify_units(name, file_date_id, ncells, rerun=0):

    paths = get_db_info()

    # exists = select_db(paths['db'], 'cell_type', '*', 'name=? AND file_date_id=?', (name, file_date_id), unique=False)
    # existing_cell_tups = execute_sql('SELECT i_cell FROM cell_type WHERE name="{}" AND file_date_id="{}" ORDER BY i_cell ASC'.format(
    #     name, file_date_id), 'my', keys=False)
    # # existing_cell_tups = execute_sql(
    # #     'SELECT i_cell FROM cell_type WHERE name="{}" AND file_date_id="{}" ORDER BY i_cell ASC'.format(
    # #         name, file_date_id), paths['db'], keys=False)
    # # existing_cells = np.array([list(x.values()) for x in existing_cell_tups]).flatten()
    # existing_cells = [x for (x,) in existing_cell_tups]
    # print(existing_cells)

    ret = select_db(paths['db'], 'ephys', '*', 'name=? AND file_date_id=?', (name, file_date_id))
    sr = ret['samp_rate']
    mid = ret['mid']
    file_date = ret['file_date']

    # if on_cluster():
    #     data_root = paths['remote_ephys_root']
    # else:
    #     data_root = '/mnt/nas/ephys/'

    data_root = paths['ephys_root']
    neuron_path = os.path.join(data_root, name, file_date_id, 'ks_matlab')
    df = pd.read_csv(os.path.join(neuron_path, 'cluster_info.tsv'), delimiter='\t')
    good_inds = df['group'] == 'good'
    good_df = df[good_inds]
    chans = good_df['ch']

    existing_cell_path = os.path.join(neuron_path, 'cell_types.p')
    print(existing_cell_path)
    if os.path.isfile(existing_cell_path) and not rerun:
    # if len(existing_cells) == ncells and not rerun:
        print('Already found classification for name = {}, file_date_id = {}. Skipping'.format(name, file_date_id))
        return good_df['cluster_id']

    templates = np.load(os.path.join(neuron_path, 'templates.npy'))
    spike_samples = np.load(os.path.join(neuron_path, 'spike_times.npy')).flatten()
    spike_templates = np.load(os.path.join(neuron_path, 'spike_templates.npy')).flatten()
    spike_clusters = np.load(os.path.join(neuron_path, 'spike_clusters.npy'))

    spike_times = spike_samples / sr

    bin_size = 1e-3  # 1 ms
    window_size = 1.8  # 900 ms in each direction
    window_center = int(window_size / bin_size / 2)
    comp_window_dur = int(0.3 / bin_size)  # indices for 600-900 ms lag in autocorrelation
    # corrs = correlograms(spike_times, spike_clusters, sample_rate=sr, bin_size=bin_size, window_size=window_size)

    with open(os.path.join(paths['config'])) as json_file:
        db_info = json.load(json_file)
    col_names = tuple([x[0] for x in db_info['cell_type_fields']])  # keys in the db
    # insert_sql = 'REPLACE INTO cell_type (' + ', '.join(col_names) + ') VALUES (' + ', '.join(
    #     ['%s'] * len(col_names)) + ')'

    all_values = []

    for i_cell, id in enumerate(good_df['cluster_id']):

        # if i_cell not in existing_cells:

        # print('Trying cell {}'.format(i_cell))
        cell_templates = spike_templates[spike_clusters == id]
        main_template_ind = mode(cell_templates, axis=None).mode[0]
        main_ch = chans.iloc[i_cell]
        template = templates[main_template_ind, :, main_ch]
        trough_samp = np.argmin(template)
        peak_samp = np.argmax(template)
        trough_to_peak_dur = (peak_samp - trough_samp) / sr
        isis = np.diff(spike_samples[spike_clusters == id].flatten()) / sr
        long_isis = np.sum(isis[isis > 2])
        frac_long_isis = long_isis / ret['recording_dur']

        # print('Computing correlograms')

        autocorr = correlograms(spike_times[spike_clusters == id], spike_clusters[spike_clusters == id], cluster_ids=[id],
                                sample_rate=sr, bin_size=bin_size, window_size=window_size)[0, 0, :]
        comp_rate = np.mean(autocorr[:comp_window_dur])
        suppressed_bins = np.argmax(autocorr[window_center:] >= comp_rate)
        suppressed_time = suppressed_bins * bin_size

        if trough_to_peak_dur < 0:
            cell_type = 'axonal'
        elif trough_to_peak_dur < 400e-6:
            if suppressed_time > .040:
                cell_type = 'TAN neurites'
            elif frac_long_isis > 0.1:
                cell_type = 'unidentified'
            else:
                cell_type = 'FSI'
        elif suppressed_time > .040:  # if post-spike suppression is > 40 ms
            cell_type = 'TAN'
        else:
            cell_type = 'MSN'

        # print('Done classifying')

        post_spike_ms = int(suppressed_time * 1e3)  # seconds to ms
        peak_to_trough_us = trough_to_peak_dur * 1e6  # seconds to microseconds

        # try:
        #     # get information from neuron_regress table if it exists there
        #     cell = select_db(paths['db'], 'neuron_regress', '*', 'name=? AND file_date_id=? AND i_cell=?',
        #                  (name, file_date_id, i_cell))
        #     for var in cell.keys():
        #         exec("{} = cell['{}']".format(var, var))
        #
        # except:

        # get what little info we can from cluster_info.tsv
        row = good_df.iloc[i_cell]  # df that is returned is all good_inds, so just get the good_ind I am analyzing
        # bring loaded variables into namespace
        int_headers = ['cluster_id', 'ch', 'n_spikes']
        for var in int_headers:
            exec("{} = int(row['{}'])".format(var, var))
        float_headers = ['Amplitude', 'amp', 'fr']
        for var in float_headers:
            exec("{} = row['{}']".format(var, var))

        # print('Retrieved info')

        # stash in cell_type table
        loc = locals()
        row_values = tuple([loc[k] if k in loc and not (isinstance(loc[k], float) and np.isnan(loc[k])) else None for k in col_names])
        all_values.append(row_values)

        # # insert_into_db('my', 'cell_type', all_keys, all_vals)
        # if (i_cell + 1) % 20 == 0:
        #     # print('Attempting to stash neurons {}-{} into cell_type for mouse {} file_date_id {}'.format(
        #     #     i_cell-19, i_cell, name, file_date_id))
        #     execute_many(insert_sql, 'my', all_values)
        #     print('Stashed neurons {}-{} into cell_type for mouse {} file_date_id {}'.format(
        #         i_cell-19, i_cell, name, file_date_id))
        #     all_values = []
            # time.sleep(0.1)

        # else:
        #     print('Found cell {} in cell_type table'.format(i_cell))

    cell_df = pd.DataFrame(all_values, columns=col_names)
    with open(existing_cell_path, 'wb') as f:
        pickle.dump(cell_df, f)
    # print(all_values)
    # if len(all_values) > 0:
    #     execute_many(insert_sql, 'my', all_values)

    return good_df['cluster_id']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Classify units')
    parser.add_argument('name', type=str)
    parser.add_argument('file_date_id', type=str)
    parser.add_argument('ncells', type=int)
    parser.add_argument('-r', '--rerun', type=int, default=0)
    args = parser.parse_args()
    classify_units(args.name, args.file_date_id, args.ncells, args.rerun)
