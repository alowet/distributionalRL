import numpy as np
import pandas as pd
import hashlib
import os
from pathlib import Path
import pickle
import json
from scipy import stats
from scipy.interpolate import interp1d
import sys
import itertools
from parallelize_correlations import parallelize_correlations
from multiprocessing import Queue, Process, Manager
import joblib
import psutil
from neuralUtils import *
from analysisUtils import *
from bg_atlasapi import BrainGlobeAtlas
sys.path.append('../utils')
from db import get_db_info, create_connection, execute_sql, select_db
from protocols import load_params
from matio import loadmat
sys.path.append('../behavior_analysis')


class Strat():
    def __init__(self, n_recorded_cells, n_bins, n_splits, split=True):
        if split:
            self.dat = np.zeros((n_recorded_cells, n_bins, 5, n_splits))
        else:
            self.dat = np.zeros((n_recorded_cells, n_bins, 5))
        self.inds = None
        self.dur = None
        self.range = None
        self.sig = None


def load_processed(table, activity_type, protocol, kwargs, rets, save_path, do_split=False, fast_load=False, only_neuron_info=False):
    """
    :param table:
    :param activity_type:
    :param protocol:
    :param kwargs:
    :param rets:
    :param save_path:
    :param do_split: whether to calculate correlations across separate splits (usually halves) of the data
    :return:
    """
    # inclusion criteria for cells
    if table == 'ephys':
        min_fr = 0.1  # Hz
        max_cv = 1
    else:
        min_fr = 0
        max_cv = np.inf

    colors, protocol_info, periods, _ = load_params(protocol)
    paths = get_db_info()
    activity_label = get_activity_label(activity_type)
    pdf_label = get_pdf_label(activity_type)

    # for plotting
    if table == 'imaging':
        n_samps = 91
        pcolor_time_full = np.linspace(-1., 5., num=n_samps + 1)
        bin_size = np.mean(np.diff(pcolor_time_full))
    else:
        bin_size = .001
        pcolor_time_full = np.arange(-1., 5 + bin_size, bin_size)
        n_samps = len(pcolor_time_full) - 1
    std_time_full = (pcolor_time_full[:-1] + pcolor_time_full[
                                             1:]) / 2  # midpoints of pcolor_time, since pcolor labels both endpoints

    # for convenience
    n_trace_types = protocol_info['n_trace_types']
    n_trial_types = protocol_info['n_trial_types']
    n_periods = periods['n_periods']
    n_prerew_periods = periods['n_prerew_periods']
    n_comp_periods = periods['n_comp_periods']
    alpha = periods['alpha']
    rew_types = np.unique(np.concatenate(protocol_info['dists']))
    rpe_types = np.unique(np.concatenate(
        [protocol_info['dists'][i_type] - protocol_info['mean'][i_type] for i_type in range(n_trial_types)]))
    combo_types = np.concatenate(
        [np.unique(protocol_info['dists'][i_type] + i_type * (1 + max(rew_types))) for i_type in range(n_trial_types)])

    # for psth
    n_splits = 2  # just split in half
    psth_bin_width = 0.25
    n_psth_bins = int(6 / psth_bin_width)
    psth_bins = np.linspace(pcolor_time_full[0], pcolor_time_full[-1], n_psth_bins + 1)

    psth_bin_centers = np.linspace(pcolor_time_full[0] + psth_bin_width / 2, pcolor_time_full[-1] - psth_bin_width / 2,
                                   n_psth_bins)
    odor_on_psth_ind = np.argmin(np.abs(psth_bin_centers - 0.125))  # from 0 - 0.25 seconds after odor delivery
    et_start_psth_ind = np.argmin(np.abs(psth_bin_centers - 1.125))  # from 1 - 1.25 seconds after odor delivery
    et_stop_psth_ind = np.argmin(np.abs(psth_bin_centers - 2.125))  # from 2-2.25 seconds after odor
    lt_psth_ind = np.argmin(np.abs(psth_bin_centers - 2.875))  # Late trace: 0.25 - 0 seconds before reward delivery
    rew_psth_ind = np.argmin(np.abs(psth_bin_centers - 3.375))  # from 0.25 - 0.5 seconds after reward delivery
    psth_pairs = ((0, odor_on_psth_ind),
                  (odor_on_psth_ind, et_start_psth_ind),
                  (et_start_psth_ind, et_stop_psth_ind),
                  (et_stop_psth_ind, lt_psth_ind + 1))  # this is analogous to idx_pairs

    offset_bins = 2  # how many bins after odor onset to wait before computing trace slice
    trace_dur_inds = (lt_psth_ind + 1 - odor_on_psth_ind - offset_bins)
    n_shuff = 500
    rng = np.random.default_rng()

    n_types_per_subdivision = [n_trial_types, len(rew_types), len(rpe_types), len(combo_types)]
    timecourses = {'cs': {}, 'rew': {}, 'rpe': {}, 'combo': {}}
    for key, n in zip(timecourses.keys(), n_types_per_subdivision):
        timecourses[key]['zF'] = np.array([], dtype=np.float64).reshape(n, 0, n_samps)
        timecourses[key]['psth'] = np.array([], dtype=np.float64).reshape(n, 0, n_psth_bins, n_splits)
        timecourses[key]['shuff'] = np.array([], dtype=np.float64).reshape(n_shuff, n, 0, n_psth_bins, n_splits)

    # # for storing data
    # timecourses = {
    #     'cs': np.array([], dtype=np.float64).reshape(n_trial_types, 0, n_samps),  # will be n_trial_types x n_neurons x n_samps
    #     'rew': np.array([], dtype=np.float64).reshape(len(rew_types), 0, n_samps),
    #     'rpe': np.array([], dtype=np.float64).reshape(len(rpe_types), 0, n_samps),
    #     'combo': np.array([], dtype=np.float64).reshape(len(combo_types), 0, n_samps)
    # }
    #
    #
    # neuron_psth = np.array([], dtype=np.float64).reshape(n_trial_types, 0, n_psth_bins, n_splits)  # 2 because it will be split between training and testing set
    # shuff_neuron_psth = np.array([], dtype=np.float64).reshape(n_shuff, n_trial_types, 0, n_psth_bins, n_splits)
    high_tt_concat = np.array([], dtype=np.float64).reshape(n_splits, 0, n_samps)

    # n_trial_types x n_neurons x n_periods
    X_means = np.array([], dtype=np.float64).reshape(n_trial_types, 0, n_periods)
    cell_stds = np.array([], dtype=np.float64).reshape(n_trial_types, 0, n_periods)
    X_shuff_means = np.array([], dtype=np.float64).reshape(n_shuff, n_trial_types, 0, n_periods)

    # don't really use these anymore, but compute them anyway
    modulation = np.array([], dtype=np.float64).reshape(0, n_comp_periods)
    discrim = np.array([], dtype=np.float64).reshape(0, n_periods)
    separated = np.array([], dtype=np.float64).reshape(0, n_periods)
    # rew_correls = np.array([], dtype=np.float64).reshape(0, n_periods, 5)  # n_neurons x n_comp_periods x 5
    # sig_values = np.array([], dtype=np.float64)

    # for storing neuron info
    neuron_info = {
        'names': [],
        'file_dates': [],
        'cluster_id': [],  # cluster id from Phy
        'neuron_idx_good': [],  # index within all cells labeled as "good" in Phy
        'neuron_idx_inc': [],  # index within subset of "good cells" that also meet ephys inclusion criteria (mean and CV)
        'fig_paths': [],
        'depths': [],
        'aps': [],
        'mls': [],
        'regions': [],
        'region_ids': [],
        'kim_regions': [],
        'kim_region_ids': [],
        'kim_generals': [],
        'means': [],
        'stds': [],
        'cvs': [],
        'kurtosis': [],
        'cell_types': [],
        'genotype': [],
        'zF_means': [],
        'zF_stds': [],
        'dFF_means': [],
        'dFF_stds': [],
        'spks_means': [],
        'spks_stds': []
    }

    rew_time = 3
    odor_on_full_ind = np.argmin(np.abs(pcolor_time_full))
    rew_on_full_ind = np.argmin(np.abs(pcolor_time_full - rew_time))  # reward onset

    n_rets = len(rets)

    kim_atlas = BrainGlobeAtlas("kim_mouse_25um", brainglobe_dir=paths['brainglobe_dir'], check_latest=False)

    # for Dabney expectile fits
    n_max_trials_per_type = 90
    cue_resps = np.array([], dtype=np.float64).reshape(n_trace_types, 0, n_max_trials_per_type,
                                                       n_prerew_periods)  # baseline, odor, early trace, late trace
    all_spk_cnts = np.array([], dtype=np.int32).reshape(n_trial_types, 0, n_max_trials_per_type, n_psth_bins)
    combo_spk_cnts = np.array([], dtype=np.int32).reshape(len(combo_types), 0, n_max_trials_per_type, n_psth_bins)
    # all_spks_ms = np.array([], dtype=np.int32).reshape(n_trial_types, 0, n_max_trials_per_type, rew_on_full_ind-odor_on_full_ind)
    mnemonic_mat_ds_trial = np.array([], dtype=np.int32).reshape(n_trace_types, 0, n_max_trials_per_type)
    mnemonic_mat_ds_combo_trial = np.array([], dtype=np.int32).reshape(len(combo_types) - 1, 0, n_max_trials_per_type)

    mids = np.zeros(n_rets, dtype=np.int16)
    cells_per_sess = np.zeros(n_rets, dtype=np.int16)
    all_sess_trial_means = [[]] * n_rets
    all_sess_binned_means = [[]] * n_rets
    all_trial_inds = [[]] * n_rets
    all_trial_types = [[]] * n_rets
    all_rewards = [[]] * n_rets
    all_tot_licks = [[]] * n_rets

    exclude_tt = protocol_info['exclude_tt']
    odor_ind = 1
    late_trace_ind = 3
    main_comp_ind = 1  # odor ind

    many_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
                   '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
                   '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#000000']

    # bin_width = 0.5 # seconds
    # bin_step = 0.1  # seconds
    # per = np.mean(np.diff(std_time_full))
    # bin_width_samples = int(round(bin_width / per))
    # trial_dur = pcolor_time_full[-1] - pcolor_time_full[0]
    # n_steps = int(round((trial_dur - bin_width)/ bin_step)) + 1
    # tile_time = np.linspace(pcolor_time_full[0] + bin_width/2, pcolor_time_full[-1] - bin_width/2, n_steps)
    # odor_step_ind = np.argmin(np.abs(tile_time - 0.5))  # from 0.25 - 0.75 seconds after odor delivery
    # lt_step_ind = np.argmin(np.abs(tile_time - 2.75)) # Late trace: 0.5 - 0 seconds before reward delivery
    # rew_step_ind = np.argmin(np.abs(tile_time - 3.5))  # from 0.25 - 0.75 seconds after reward delivery

    # load data
    for i_ret, ret in enumerate(rets):

        # convert database return value from tuple to dictionary
        mids[i_ret] = ret['mid']

        # load processed data from that behavior session
        if table == 'ephys':
            fname = [ret['name'], str(ret['file_date']), 'spikes']
        elif table == 'imaging':
            fname = [ret['name'], str(ret['file_date']), 'Ca']
        fname = '_'.join(fname) + '.p'
        fpath = os.path.join(ret['figure_path'], fname).replace(paths['remote_neural_fig_root'], paths['neural_fig_roots'][0])

        print(fpath)
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        # for concision
        activity = data[activity_type]
        all_trial_type_inds = data[
            'trial_inds_all_types']  # this includes unexpected reward trials. Formatted as a list of len(n_trial_types)
        all_trial_inds[i_ret] = all_trial_type_inds.copy()
        # trial_type_inds = [x for i, x in enumerate(all_trial_type_inds) if i not in exclude_tt]  # exclude unexpected
        # all_trial_inds[i_ret] = trial_type_inds.copy()
        timestamps = data['timestamps']
        times = timestamps['time']

        try:
            bret = select_db(paths['db'], 'session', '*', 'name=? AND exp_date=? AND (has_ephys=1 OR has_imaging=1) AND quality>=2', (ret['name'], ret['file_date']))
        except:
            brets = select_db(paths['db'], 'session', '*','name=? AND exp_date=? AND (has_ephys=1 OR has_imaging=1) AND quality>=2', (ret['name'], ret['file_date']), unique=False)
            exp_times = np.array([bret['exp_time'] for bret in brets])
            bret_ind = np.argmin(np.abs(exp_times - ret['meta_time'] * 100))
            bret = brets[bret_ind]

        bpath = os.path.join(bret['figure_path'], '_'.join([ret['name'], protocol, str(bret['exp_date']), str(bret['exp_time']).zfill(6) + '.p']))
        bpath = bpath.replace(paths['remote_behavior_fig_root'], paths['behavior_fig_roots'][0])
        # print(bpath)
        with open(bpath, 'rb') as f:
            bdata = pickle.load(f)

        # bregexp = os.path.join(paths['behavior_fig_roots'][0], ret['name'], ret['file_date_id'], '*.p')
        # bfiles = glob.glob(bregexp)
        # if len(bfiles) == 1:
        #     with open(bfiles[0], 'rb') as f:
        #         bdata = pickle.load(f)
        # else:
        #     raise Exception(
        #         'Too many behavior files for this mouse/day. Figure out a way to specify which is the recording session')

        lick_start_idx = np.argmin(np.abs(bdata['time'] - timestamps['stim'] - 1))
        lick_end_idx = np.argmin(np.abs(bdata['time'] - timestamps['stim'] - timestamps['trace']))
        ntu = ret['n_trials_used'] if isinstance(ret['n_trials_used'], int) else bret['n_trial']
        # print(ntu)
        tot_licks = np.sum(bdata['licks_raw'][:ntu, lick_start_idx:lick_end_idx], axis=1)
        all_tot_licks[i_ret] = tot_licks

        if table == 'ephys':
            trial_types = data['trial_types']  # Array of length n_trials, includes unexpected reward trials
            inc_cells = np.logical_and(data['means'] > min_fr, data['cvs'] < max_cv)  # cells to include
            neuron_info, good_df = parse_tsv(ret, neuron_info, inc_cells, kim_atlas)
            neuron_info['cluster_id'].extend(good_df['cluster_id'].iloc[inc_cells])
            neuron_info['neuron_idx_good'].extend(good_df.index[inc_cells])
            neuron_info['means'].extend(data['means'][inc_cells])
            neuron_info['stds'].extend(data['stds'][inc_cells])
            neuron_info['cvs'].extend(data['cvs'][inc_cells])
            neuron_info['kurtosis'].extend(data['kurtosis'][inc_cells])

            # conn = create_connection(paths['db'])
            if on_cluster():
                conn = create_connection('my')
                df = pd.read_sql('SELECT i_cell, cell_type FROM cell_type WHERE name="{}" AND file_date_id="{}"'.format(
                    ret['name'], ret['file_date_id']), conn, index_col='i_cell')
                conn.close()
                # print(df, inc_cells.shape, inc_cells)
                neuron_info['cell_types'].extend(df['cell_type'].iloc[inc_cells].values)
        else:
            # Array of length n_trials, includes unexpected reward trials
            trial_types = np.zeros(np.concatenate([x for x in data['trial_inds_all_types']]).max() + 1, dtype=int)
            for i_type in data['active_types']:
                trial_types[data['trial_inds_all_types'][i_type]] = i_type
            inc_cells = np.ones(np.shape(activity)[0], dtype=bool)
            # print(data['F'].shape)
            neuron_info['zF_means'].extend(np.nanmean(data['F'], axis=(1, 2)))
            neuron_info['zF_stds'].extend(np.nanstd(data['F'], axis=(1, 2)))
            neuron_info['dFF_means'].extend(np.nanmean(data['dFF'], axis=(1, 2)))
            neuron_info['dFF_stds'].extend(np.nanstd(data['dFF'], axis=(1, 2)))
            neuron_info['spks_means'].extend(np.nanmean(data['spks'], axis=(1, 2)))
            neuron_info['spks_stds'].extend(np.nanstd(data['spks'], axis=(1, 2)))
            conn = None

        all_trial_types[i_ret] = trial_types
        n_cells = np.sum(inc_cells)
        print('Total cells: {}; included cells: {}'.format(np.shape(activity)[0], n_cells))
        cells_per_sess[i_ret] = n_cells
        n_trials = np.shape(activity)[1]

        # print(len(np.concatenate(data['trial_inds_all_types'])), len(np.concatenate(trial_type_inds)), n_trials)

        # rather than a list, have a vector with the type of each trial. This EXCLUDES unexpected reward trials!
        # these_trial_types = np.zeros(n_trials, dtype=np.int16)
        # for i, x in enumerate(all_trial_type_inds):
        #     these_trial_types[x] = i
        # all_trial_types[i_ret] = these_trial_types

        neuron_info['names'].extend([ret['name']] * n_cells)
        neuron_info['file_dates'].extend([ret['file_date']] * n_cells)
        #     neuron_info['neuron_idx'].extend(np.arange(n_cells))
        neuron_info['neuron_idx_inc'].extend(np.flatnonzero(inc_cells))
        neuron_info['fig_paths'].extend([ret['figure_path']] * n_cells)
        neuron_info['genotype'].extend([ret['genotype'].split()[0]] * n_cells)
        neuron_info = {k: neuron_info[k] for k in neuron_info.keys() if len(neuron_info[k])}

        if only_neuron_info: continue

        if activity_type == 'firing':  # z-score the firing rates here, if desired.
            zF = (activity[inc_cells] - data['means'][inc_cells, np.newaxis, np.newaxis]) / data['stds'][
                inc_cells, np.newaxis, np.newaxis]
        else:
            # imaging timebase used can vary slightly from session to session depending on exact
            # sample rate, so align everything to a common timebase here
            f = interp1d(times, activity[inc_cells], 'nearest', fill_value='extrapolate', axis=2)
            # zF not just z-scored fluorescence anymore, but any kind of activity that we load
            # in depending on activity_label
            zF = f(std_time_full)
        # print(zF.shape, times.shape, std_time_full.shape)

        odor_on_idx = np.argmin(np.abs(times))
        trace_start_idx = np.argmin(np.abs(times - timestamps['stim']))
        trace_plus_1_idx = np.argmin(np.abs(times - timestamps['stim'] - 1))
        trace_end_idx = np.argmin(np.abs(times - timestamps['stim'] - timestamps['trace']))
        # go up to 1 second after reward delivery
        reward_end_idx = np.argmin(np.abs(times - timestamps['stim'] - timestamps['trace'] - 1))
        # these should all be a second in duration, otherwise I need to change the np.sum step in cases of
        # activity_type == 'spks' to a np.mean divided by the ratio of the bin width to the duration
        idx_pairs = [(0, odor_on_idx), (odor_on_idx, trace_start_idx), (trace_start_idx, trace_plus_1_idx),
                     (trace_plus_1_idx, trace_end_idx), (trace_end_idx, reward_end_idx)]

        behavior_path = ret['behavior_path'].replace(paths['remote_behavior_root'], paths['behavior_root'])
        bdata = loadmat(behavior_path)
        rewards = bdata['SessionData']['RewardDelivered'][:n_trials]
        all_rewards[i_ret] = rewards

        #     if np.sum(np.isnan(trial_avg_zF)) > 0:
        if np.any(all_trial_type_inds == []):
            print(i_ret, all_trial_type_inds)  # ignore that session if it doesn't have at least one trial of each type

        else:
            rpes = rewards - np.array(protocol_info['mean'])[trial_types]
            combos = np.zeros(n_trials, dtype=np.int16)
            for i_type in range(n_trial_types):
                for rew in rew_types:
                    use_inds = np.logical_and(rewards == rew, trial_types == i_type)
                    combos[use_inds] = rew + i_type * (1 + max(rew_types))

            spks = data['spks'][inc_cells]
            bin_inds = np.digitize(times, psth_bins) - 1
            # NOTE: IT'S VERY IMPORTANT TO HAVE THESE AS SPIKE COUNTS FOR POISSON DECODING!
            binned_spk_cnts = np.stack([np.sum(spks[..., ind == bin_inds], axis=2) for ind in np.arange(n_psth_bins)],
                                       axis=2)
            binned_spks = binned_spk_cnts / psth_bin_width

            # average by CS, reward size, RPE (actual reward minus expected value), and a combo of the two
            for timecourse, trial_vec, these_types in zip(timecourses.keys(), [trial_types, rewards, rpes, combos],
                                                          [np.arange(n_trial_types), rew_types, rpe_types, combo_types]):
                # these_types = np.unique(trial_vec)
                these_type_inds = [np.flatnonzero(trial_vec == this_type) for this_type in these_types]
                type_avg_zF = np.array(
                    [np.nanmean(zF[:, these_type_inds[i_type], :], axis=1) for i_type in range(len(these_types))])
                timecourses[timecourse]['zF'] = np.concatenate((timecourses[timecourse]['zF'], type_avg_zF), axis=1)

                perm_inds = [rng.permutation(this_type_inds) for this_type_inds in these_type_inds]
                n = len(these_types)
                split_spks = np.zeros((n, n_cells, n_psth_bins, n_splits))
                shuff_spks = np.zeros((n_shuff, n, n_cells, n_psth_bins, n_splits))
                for i_split in range(n_splits):
                    split_spks[..., i_split] = np.array(
                        [np.mean(binned_spks[:, perm_inds[i_type][i_split::n_splits], :], axis=1) for i_type in
                         range(n)])
                    for i_shuff in range(n_shuff):
                        shuff_order = rng.permutation(trial_vec)
                        shuff_types = [np.flatnonzero(shuff_order == this_type) for this_type in these_types]
                        shuff_spks[i_shuff, ..., i_split] = np.array(
                            [np.mean(binned_spks[:, shuff_types[i_type][i_split::n_splits], :], axis=1) for i_type in
                             range(n)])
                timecourses[timecourse]['psth'] = np.concatenate((timecourses[timecourse]['psth'], split_spks), axis=1)
                timecourses[timecourse]['shuff'] = np.concatenate((timecourses[timecourse]['shuff'], shuff_spks),
                                                                  axis=2)

            if fast_load:
                del spks, shuff_spks, zF
                continue

            # split high_tt into two halves, so I can sort on one half only and plot the other half
            # high_tt_timecourse = np.zeros((n_cells, n_samps, n_splits))
            # for i_split in range(n_splits):
            perm_inds = [rng.permutation(this_type_inds) for this_type_inds in all_trial_type_inds]
            high_tt_timecourse = np.array([np.nanmean(zF[:, perm_inds[protocol_info['high_tt'][-1]][i_split::n_splits], :],
                                                      axis=1) for i_split in range(n_splits)])
            high_tt_concat = np.concatenate((high_tt_concat, high_tt_timecourse), axis=1)

            trial_spk_cnts = np.full((n_trial_types, n_cells, n_max_trials_per_type, n_psth_bins), np.nan)
            ms_spks = np.full((n_trial_types, n_cells, n_max_trials_per_type, rew_on_full_ind - odor_on_full_ind),
                              np.nan)
            for i_type in range(n_trial_types):
                trial_spk_cnts[i_type, :, :len(all_trial_type_inds[i_type]), :] = binned_spk_cnts[:,
                                                                                  all_trial_type_inds[i_type], :]
                ms_spks[i_type, :, :len(all_trial_type_inds[i_type]), :] = spks[:, all_trial_type_inds[i_type],
                                                                           odor_on_full_ind:rew_on_full_ind]
            all_spk_cnts = np.concatenate((all_spk_cnts, trial_spk_cnts), axis=1)

            trial_spk_combo_cnts = np.full((len(combo_types), n_cells, n_max_trials_per_type, n_psth_bins), np.nan)
            # ms_combo_spks = np.full(
            #     (len(combo_types), n_cells, n_max_trials_per_type, rew_on_full_ind - odor_on_full_ind), np.nan)
            # for i_combo, combo in enumerate(combo_types):
            #     trial_spk_combo_cnts[i_combo, :, :np.sum(combos == combo), :] = binned_spk_cnts[:, combos == combo, :]
            #     ms_combo_spks[i_combo, :, :np.sum(combos == combo), :] = spks[:, combos == combo,
            #                                                              odor_on_full_ind:rew_on_full_ind]
            # combo_spk_cnts = np.concatenate((combo_spk_cnts, trial_spk_combo_cnts), axis=1)

            # save memory (and therefore time) by not doing combo ms
            for i_div, (n_types, these_spks) in enumerate(zip([n_trace_types], [ms_spks])):
            # for i_div, (n_types, these_spks) in enumerate(
            #         zip([n_trace_types, len(combo_types) - 1], [ms_spks, ms_combo_spks])):

                if bin_size > psth_bin_width / 10:  # will be the case for imaging sessions
                    rebin = psth_bin_width / 10
                    f_interp = interp1d(std_time_full[odor_on_full_ind:rew_on_full_ind], these_spks, axis=3,
                                        fill_value='extrapolate')
                    interp_spks = f_interp(np.arange(0, rew_time + rebin, rebin))
                else:  # will be the case for ephys sessions
                    rebin = bin_size
                    interp_spks = these_spks

                random_bins = np.arange(int(np.ceil((offset_bins * psth_bin_width) / rebin)),
                                        interp_spks.shape[3] - trace_dur_inds + 1, trace_dur_inds).reshape(1, 1, 1,
                                                                                                           -1) + \
                              rng.choice(trace_dur_inds, size=(n_types, n_cells, n_max_trials_per_type, 1))

                mnemonic_mat_ds_inc = np.sum(
                    np.take_along_axis(interp_spks[:n_types, :], random_bins, axis=3), axis=3) / (
                                              random_bins.shape[3] * rebin)

                if i_div == 0:
                    mnemonic_mat_ds_trial = np.concatenate((mnemonic_mat_ds_trial, mnemonic_mat_ds_inc), axis=1)
                else:
                    mnemonic_mat_ds_combo_trial = np.concatenate((mnemonic_mat_ds_combo_trial, mnemonic_mat_ds_inc),
                                                                 axis=1)

            # delete these (very large) variables from memory
            del spks, these_spks, interp_spks, random_bins, shuff_spks

            # random_bins = np.arange(int((offset_bins * psth_bin_width) / bin_size), ms_spks.shape[3],
            #                         trace_dur_inds).reshape(1, 1, 1, -1) + \
            #               rng.choice(trace_dur_inds, size=(n_trace_types, n_cells, n_max_trials_per_type, 1))
            # random_combo_bins = np.arange(int((offset_bins * psth_bin_width) / bin_size), ms_combo_spks.shape[3],
            #                         trace_dur_inds).reshape(1, 1, 1, -1) + \
            #               rng.choice(trace_dur_inds, size=(len(combo_types)-1, n_cells, n_max_trials_per_type, 1))
            #
            # mnemonic_mat_ds_inc = np.sum(
            #     np.take_along_axis(ms_spks[:n_trace_types, :], random_bins, axis=3), axis=3) / (
            #                                     random_bins.shape[3] * bin_size)
            # mnemonic_mat_ds_combo_inc = np.sum(
            #     np.take_along_axis(ms_combo_spks[:len(combo_types)-1, :], random_bins, axis=3), axis=3) / (
            #                                     random_bins.shape[3] * bin_size)
            # mnemonic_mat_ds_trial = np.concatenate((mnemonic_mat_ds_trial, mnemonic_mat_ds_inc), axis=1)


            # get means within windows of that trial-averaged trace, producing arrays n_trial_types x n_cells
            this_sess_means = np.zeros((n_trial_types, n_cells, n_periods))
            this_sess_stds = np.zeros((n_trial_types, n_cells, n_periods))
            this_sess_shuff_means = np.zeros((n_shuff, n_trial_types, n_cells, n_periods))

            # also get means within windows of not-trial-averaged trace, one for each cell and trial
            this_sess_trial_means = np.zeros((n_cells, n_trials, n_periods))
            for i_pair, pair in enumerate(idx_pairs):
                if activity_type == 'spks':
                    this_sess_trial_means[..., i_pair] = np.sum(zF[:, :, pair[0]:pair[1]], axis=2)
                else:
                    this_sess_trial_means[..., i_pair] = np.mean(zF[:, :, pair[0]:pair[1]], axis=2)
                #             this_sess_means[..., i_pair] = np.mean(trial_avg_zF[:, :, pair[0]:pair[1]], axis=2)
                this_sess_means[..., i_pair] = np.array(
                    [np.mean(this_sess_trial_means[:, all_trial_type_inds[i_type], i_pair],
                             axis=1) for i_type in range(n_trial_types)])
                this_sess_stds[..., i_pair] = np.array(
                    [np.std(this_sess_trial_means[:, all_trial_type_inds[i_type], i_pair],
                            axis=1) for i_type in range(n_trial_types)])

                for i_shuff in range(n_shuff):
                    this_sess_shuff_means[i_shuff, ..., i_pair] = np.array([np.nanmean(  # nanmean b/c of imaging unexpected reward trials
                        this_sess_trial_means[:, rng.permutation(trial_types) == i_type, i_pair], axis=1) for i_type in range(n_trial_types)])

            del zF

            # similar to this_sess_trial_means, get means within 500 ms windows of full trace
            # tile_means = np.zeros((n_cells, n_trials, n_steps))
            # for i_step in range(n_steps):
            #     start_idx = int(round(bin_step * i_step / per))
            #     end_idx = start_idx + bin_width_samples
            #     tile_means[:, :, i_step] = np.sum(spks[:, :, start_idx:end_idx], axis=2) / bin_width

            # save for normalization later. will contain NaNs due to inclusion of unexpected reward trials!
            all_sess_trial_means[i_ret] = np.copy(this_sess_trial_means)
            # all_sess_tile_means[i_ret] = np.copy(tile_means)
            all_sess_binned_means[i_ret] = np.copy(binned_spks)
            #         all_sess_timecourse[i_ret] = np.copy(zF)

            # main_comp_ind will always be before reward, so restrict the to n_trace_types (i.e. exclude unexpected)
            this_sess_all_types_per_means = np.full((n_trace_types, n_cells, n_max_trials_per_type, n_prerew_periods),
                                                    np.nan)
            for i_type in range(n_trace_types):
                for i_per in range(n_prerew_periods):
                    this_sess_all_types_per_means[i_type, :, :len(all_trial_type_inds[i_type]), i_per] = \
                        this_sess_trial_means[:, all_trial_type_inds[i_type], i_per]
            cue_resps = np.concatenate((cue_resps, this_sess_all_types_per_means), axis=1)

            X_means = np.concatenate((X_means, this_sess_means), axis=1)
            cell_stds = np.concatenate((cell_stds, this_sess_stds), axis=1)
            X_shuff_means = np.concatenate((X_shuff_means, this_sess_shuff_means), axis=2)

            # check for reliable mean neural activity differences vs. baseline, irrespective of trial type
            mod = np.zeros((n_cells, n_comp_periods))
            for i_period in range(1, n_periods):
                _, mod[:, i_period - 1] = stats.ttest_rel(this_sess_trial_means[..., 0],
                                                          this_sess_trial_means[..., i_period], axis=1)
            modulation = np.concatenate((modulation, mod), axis=0)

            # check for reliable activity differences across trial types
            tt_period_mean = [this_sess_trial_means[:, tt_inds, :] for tt_inds in all_trial_type_inds]
            _, disc = stats.f_oneway(*tt_period_mean, axis=1)  # different number of trials for each trial type
            discrim = np.concatenate((discrim, disc), axis=0)

            # check for reliable activity differences between lowest and highest CSs
            sep = np.zeros((n_cells, n_periods))
            #         for i_period in range(n_periods):
            #             for i_cell in range(n_cells):
            #                 _, sep[i_cell, i_period] = stats.mannwhitneyu(tt_period_mean[0][i_cell, :, i_period], tt_period_mean[-1][i_cell, :, i_period])
            for i_period in range(n_periods):
                _, sep[:, i_period] = stats.ttest_ind(tt_period_mean[0][..., i_period],
                                                      tt_period_mean[-1][..., i_period],
                                                      axis=1)
            separated = np.concatenate((separated, sep), axis=0)

            # check if it's in the neuron_regress table and significant
            # query = execute_sql(
            #     'SELECT * FROM neuron_regress WHERE name="{}" AND file_date="{}" ORDER BY i_cell ASC'.format(
            #         ret['name'], ret['file_date']), paths['db'])
            # these_sig_values = np.array([x['sig_value'] for x in query])
            # sig_values = np.concatenate((sig_values, these_sig_values))

        # get variance of these means across cells for each trial type
    #         X_vars[:, i_ret, :] = np.var(this_sess_means, axis=1)

    if only_neuron_info:
        for key, val in zip(neuron_info.keys(), neuron_info.values()):
            if val and type(val[0]) == str:
                neuron_info[key] = np.array(val, dtype='object')
            else:
                neuron_info[key] = np.array(val)
        with open(save_path, 'wb') as f:
            joblib.dump(neuron_info, f)
        return neuron_info

    if not fast_load:
        cue_spk_cnts = np.stack([np.sum(all_spk_cnts[:n_trace_types, :, :, st:stop], axis=-1) for (st, stop) in psth_pairs],
                                axis=-1)

        # for enforcing consistency across trials, use coefficient of variation for now
        # max_covar = 0.1
        # # also enforce minimum avg (absolute) activity
        # min_firing = 0.1

        # cv = cell_stds/X_means
        # cell_mask = np.logical_and(cv < max_covar, np.abs(X_means) > min_firing)

        # make this all true for now, because I computed coefficient of variation across entire recording session,
        # not across trials
        # cell_mask = np.ones((n_trial_types, np.sum(cells_per_sess), n_periods), dtype=bool)

        # for difference
        min_diff = 0.2
        # diff_mask = np.abs(X_means[-1, :, main_comp_ind] - )

        # have a separate loop down here so I can preallocate, for speed
        n_recorded_cells = X_means.shape[1]

        # trace_dur_corrs = {}
        corrs = {}
        corrs_seconds = {}
        # manager = Manager()
        # trace_dur_corrs = manager.dict()
        # corrs = manager.dict()

        prerew_keys = ['mean', 'var', 'cvar', 'resid_mean', 'resid_var', 'resid_cvar']
        postrew_keys = ['rew', 'rpe']
        corr_keys = prerew_keys + postrew_keys

        for use_corrs, n_bins in zip([corrs, corrs_seconds], [n_psth_bins, n_periods]):
            for key in corr_keys + ['nolick_mean']:
                use_corrs[key] = {}
                # trace_dur_corrs[key] = np.zeros((n_recorded_cells, 5))
                # corrs[key] = manager.dict()
                for order in ['ord', 'scram']:
                    use_corrs[key][order] = {}
                    # corrs[key][order] = manager.dict()
                    use_corrs[key][order]['all'] = Strat(n_recorded_cells, n_bins, n_splits, split=False)
                    if do_split: use_corrs[key][order]['split'] = Strat(n_recorded_cells, n_bins, n_splits)

        scrambled_cue_resps = np.full(cue_resps.shape, np.nan)
        scrambled_cue_spk_cnts = np.full(cue_spk_cnts.shape, np.nan)
        all_scrambled_order = np.zeros((n_trace_types, n_recorded_cells))
        all_binned_minus_mean = [[]] * n_rets

        start_cell = 0

        # this correlation analysis will apply only to expected reward trials (i.e. included), therefore n_trace_types
        for i_ret, ret in enumerate(rets):
            print(ret['name'], ret['file_date_id'])
            end_cell = start_cell + cells_per_sess[i_ret]

            # trialwise correlation of expected value
            inds = all_trial_inds[i_ret]
            trial_types = all_trial_types[i_ret]
            # tiled = all_sess_tile_means[i_ret]
            binned = all_sess_binned_means[i_ret]
            binned_seconds = all_sess_trial_means[i_ret]
            # binned_minus_mean = np.zeros(binned.shape)
            trace_inds = trial_types < n_trace_types

            # bret = select_db(paths['db'], 'session', '*', 'name=? AND exp_date=? AND (has_ephys=1 OR has_imaging=1) AND quality>=2', (ret['name'], ret['file_date']))
            # ntu = ret['n_trials_used'] if 'n_trials_used' in ret else bret['n_trial']
            # print(trace_inds.shape)
            # print((all_tot_licks[i_ret] == 0).shape)
            nolick_inds = np.logical_and(trace_inds, all_tot_licks[i_ret] == 0)
            # trace_mean = np.mean(binned[:, trace_inds, odor_on_psth_ind + offset_bins:lt_psth_ind + 1, np.newaxis], axis=2)
            # mean is okay here because binned is a rate already
            # late_trace_mean = np.mean(binned[:, trace_inds, et_stop_psth_ind:lt_psth_ind + 1, np.newaxis], axis=2)
            rewards = all_rewards[i_ret]

            # p = Process(target=parallelize_correlations, args=(start_cell, end_cell, inds, trial_types, binned, trace_inds,
            #                                                    trace_mean, corrs, trace_dur_corrs, prerew_keys,
            #                                                    postrew_keys, protocol_info, rewards, cue_resps,
            #                                                    cue_spk_cnts, all_scrambled_order, scrambled_cue_resps,
            #                                                    scrambled_cue_spk_cnts, n_splits))
            # p.start()
            # p.join()

            # parallelize_correlations(queue, start_cell, end_cell, inds, trial_types, binned, trace_inds, trace_mean, corrs,
            #     trace_dur_corrs, prerew_keys, postrew_keys, protocol_info, rewards, cue_resps,
            #     cue_spk_cnts, all_scrambled_order, scrambled_cue_resps, scrambled_cue_spk_cnts, n_splits)

            # correls without train/test split
            for use_corrs, use_binned in zip([corrs, corrs_seconds], [binned, binned_seconds]):
                for key in corr_keys:
                    if key in prerew_keys:
                        full_stat = np.array(protocol_info[key])[trial_types[trace_inds]]
                        # trace_dur_corrs[key][start_cell:end_cell, :] = find_trial_rew_correls(late_trace_mean, full_stat)[:, 0, :]
                    elif key == 'rew':
                        full_stat = rewards[trace_inds]
                    elif key == 'rpe':
                        full_stat = rewards[trace_inds] - np.array(protocol_info['mean'])[trial_types[trace_inds]]

                    if key == 'resid_mean':
                        bin_var_preds = use_corrs['var']['ord']['all'].dat[start_cell:end_cell, np.newaxis, :, 1] + \
                                          use_corrs['var']['ord']['all'].dat[start_cell:end_cell, np.newaxis, :, 0] * \
                                          np.array(protocol_info['var'])[np.newaxis, :, np.newaxis]
                        binned_minus_var = use_binned - bin_var_preds[:, trial_types, :]
                        active_binned = binned_minus_var
                        # all_binned_minus_var[i_ret] = binned_minus_var

                    elif key == 'resid_var' or key == 'resid_cvar':
                        # project out the value regression prediction for all 250 ms bins
                        bin_value_preds = use_corrs['mean']['ord']['all'].dat[start_cell:end_cell, np.newaxis, :, 1] + \
                                          use_corrs['mean']['ord']['all'].dat[start_cell:end_cell, np.newaxis, :, 0] * \
                                          np.array(protocol_info['mean'])[np.newaxis, :, np.newaxis]
                        binned_minus_mean = use_binned - bin_value_preds[:, trial_types, :]
                        active_binned = binned_minus_mean
                        all_binned_minus_mean[i_ret] = binned_minus_mean

                    else:
                        active_binned = use_binned
                    # print(active_binned.shape)
                    use_corrs[key]['ord']['all'].dat[start_cell:end_cell, :, :] = find_trial_rew_correls(active_binned[:, trace_inds, :], full_stat)
                    if key == 'mean':
                        use_corrs['nolick_mean']['ord']['all'].dat[start_cell:end_cell, :, :] = find_trial_rew_correls(
                            active_binned[:, nolick_inds, :], np.array(protocol_info['mean'])[trial_types[nolick_inds]])

                scrambled_order = np.array([rng.permutation(x) for x in
                                            np.tile(np.arange(n_trace_types)[np.newaxis, :], [cells_per_sess[i_ret], 1])]).T
                all_scrambled_order[:, start_cell:end_cell] = scrambled_order
                scrambled_cue_resps[:, start_cell:end_cell] = np.take_along_axis(cue_resps[:, start_cell:end_cell],
                                                                                 scrambled_order[..., np.newaxis, np.newaxis],
                                                                                 axis=0)
                scrambled_cue_spk_cnts[:, start_cell:end_cell] = np.take_along_axis(cue_spk_cnts[:, start_cell:end_cell],
                                                                                    scrambled_order[
                                                                                        ..., np.newaxis, np.newaxis], axis=0)
                # shuffle odor identities to see what comes out
                scrambled_inds = np.array(inds, dtype='object')[scrambled_order]
                for i_cell in range(cells_per_sess[i_ret]):

                    # note that all_scrambled_activity is grouped by (scrambled) trial type. So all (scrambled) type 0 trials
                    # come first, then (scrambled) type 1, etc. All the scrambled stat vectors should adhere to the same
                    # convention, as should resid_var
                    all_scrambled_activity = np.concatenate([use_binned[i_cell, scrambled_inds[i_type, i_cell], :] for i_type in range(n_trace_types)], axis=0)
                    # all_scrambled_minus_mean = np.concatenate(
                    #     [binned_minus_mean[i_cell, scrambled_inds[i_type, i_cell], :] for i_type in range(n_trace_types)], axis=0)
                    n_trials_per_scrambled_type = [len(scrambled_inds[i_type, i_cell]) for i_type in range(n_trace_types)]

                    for key in corr_keys:
                        # the point of "scrambled stat vec" is not to actually scramble the order (we already did that to
                        # the activity), but rather to get the number of trials correct
                        if key in prerew_keys:
                            scrambled_stat_vec = np.concatenate([[protocol_info[key][i]] * x for i, x in enumerate(n_trials_per_scrambled_type)])
                        elif key == 'rew':
                            scrambled_stat_vec = np.concatenate([rng.choice(protocol_info['dists'][i], size=x) for i, x in enumerate(n_trials_per_scrambled_type)])
                                # [rewards[scrambled_inds[i_type, i_cell]] for i_type in range(n_trace_types)])
                        elif key == 'rpe':
                            scram_rew_vec = np.concatenate([rng.choice(protocol_info['dists'][i], size=x) for i, x in enumerate(n_trials_per_scrambled_type)])
                                # [rewards[scrambled_inds[i_type, i_cell]] for i_type in range(n_trace_types)])
                            scram_mean_vec = np.concatenate([[protocol_info['mean'][i]] * x for i, x in enumerate(n_trials_per_scrambled_type)])
                            scrambled_stat_vec = scram_rew_vec - scram_mean_vec

                        if key == 'resid_mean':
                            # project out the variance regression prediction for all 250 ms bins
                            bin_scram_var_preds = use_corrs['var']['scram']['all'].dat[start_cell + i_cell, np.newaxis, :, 1] + \
                                                    use_corrs['var']['scram']['all'].dat[start_cell + i_cell, np.newaxis, :, 0] * \
                                                    np.array(protocol_info['var'])[np.newaxis, :n_trace_types, np.newaxis]
                            use_all_scrambled = all_scrambled_activity - np.squeeze(np.repeat(bin_scram_var_preds, n_trials_per_scrambled_type, axis=1))

                        if key == 'resid_var' or key == 'resid_cvar':
                            # project out the value regression prediction for all 250 ms bins
                            bin_scram_value_preds = use_corrs['mean']['scram']['all'].dat[start_cell + i_cell, np.newaxis, :, 1] + \
                                                    use_corrs['mean']['scram']['all'].dat[start_cell + i_cell, np.newaxis, :, 0] * \
                                                    np.array(protocol_info['mean'])[np.newaxis, :n_trace_types, np.newaxis]
                                                    # np.array(protocol_info['mean'])[np.newaxis, scrambled_order[:, i_cell], np.newaxis]
                            use_all_scrambled = all_scrambled_activity - np.squeeze(np.repeat(bin_scram_value_preds, n_trials_per_scrambled_type, axis=1))
                        else:
                            use_all_scrambled = all_scrambled_activity

                        # use_all_scrambled = all_scrambled_minus_mean if key == 'resid_var' else all_scrambled_activity

                        use_corrs[key]['scram']['all'].dat[start_cell + i_cell, :, :] = find_trial_rew_correls(
                            use_all_scrambled[np.newaxis, ...], scrambled_stat_vec)
                        if key == 'mean':
                            nolick_scram_inds = [scrambled_inds[i_type, i_cell][
                                                     np.isin(scrambled_inds[i_type, i_cell], np.flatnonzero(nolick_inds))]
                                                 for i_type in range(n_trace_types)]
                            nolick_scrambled_activity = np.concatenate(
                                [use_binned[i_cell, nolick_scram_inds[i_type], :] for i_type in range(n_trace_types)], axis=0)
                            n_nolick_trials_per_scrambled_type = [len(nolick_scram_inds[i_type]) for i_type in range(n_trace_types)]
                            use_corrs['nolick_mean']['scram']['all'].dat[start_cell + i_cell, :, :] = find_trial_rew_correls(
                                nolick_scrambled_activity[np.newaxis, ...], np.concatenate(
                                    [[protocol_info['mean'][i]] * x for i, x in enumerate(n_nolick_trials_per_scrambled_type)]))

                        if do_split:

                            for i_split in range(n_splits):

                                scrambled_activity = np.concatenate(
                                    [np.array(use_binned[i_cell, scrambled_inds[i_type, i_cell][i_split::n_splits], :]) for i_type in
                                     range(n_trace_types)], axis=0)
                                # scrambled_minus_mean = np.concatenate(
                                #     [np.array(binned_minus_mean[i_cell, scrambled_inds[i_type, i_cell][i_split::n_splits], :]) for
                                #      i_type in range(n_trace_types)], axis=0)

                                n_split_trials_per_scrambled_type = [len(scrambled_inds[i_type, i_cell][i_split::n_splits])
                                                                     for i_type in range(n_trace_types)]

                                if key in prerew_keys:
                                    split_scrambled_stat_vec = np.concatenate(
                                        [[protocol_info[key][i]] * x for i, x in enumerate(n_split_trials_per_scrambled_type)])
                                elif key == 'rew':
                                    split_scrambled_stat_vec = np.concatenate([rng.choice(protocol_info['dists'][i], size=x) for i, x in enumerate(n_split_trials_per_scrambled_type)])
                                        # [rewards[scrambled_inds[i_type, i_cell][i_split::n_splits]] for i_type in range(n_trace_types)])
                                elif key == 'rpe':
                                    # need to repeat this because of the splits
                                    split_scram_rew_vec = np.concatenate([rng.choice(protocol_info['dists'][i], size=x) for i, x in enumerate(n_split_trials_per_scrambled_type)])
                                        # [rewards[scrambled_inds[i_type, i_cell][i_split::n_splits]] for i_type in range(n_trace_types)])
                                    split_scram_mean_vec = np.concatenate(
                                        [[protocol_info['mean'][i]] * x for i, x in enumerate(n_split_trials_per_scrambled_type)])
                                    split_scrambled_stat_vec = split_scram_rew_vec - split_scram_mean_vec

                                if key == 'resid_mean':
                                    # project out the variance regression prediction for all 250 ms bins
                                    bin_scram_split_var_preds = use_corrs['var']['scram']['split'].dat[start_cell + i_cell, np.newaxis, :, 1, i_split] + \
                                                            use_corrs['var']['scram']['split'].dat[start_cell + i_cell, np.newaxis, :, 0, i_split] * \
                                                            np.array(protocol_info['var'])[np.newaxis, :n_trace_types, np.newaxis]
                                    use_scrambled = scrambled_activity - np.squeeze(np.repeat(bin_scram_split_var_preds, n_split_trials_per_scrambled_type, axis=1))
                                elif key == 'resid_var' or key == 'resid_cvar':
                                    # project out the value regression prediction for all 250 ms bins
                                    bin_scram_split_mean_preds = use_corrs['mean']['scram']['split'].dat[start_cell + i_cell, np.newaxis, :, 1, i_split] + \
                                                            use_corrs['mean']['scram']['split'].dat[start_cell + i_cell, np.newaxis, :, 0, i_split] * \
                                                            np.array(protocol_info['mean'])[np.newaxis, :n_trace_types, np.newaxis]
                                                            # np.array(protocol_info['mean'])[np.newaxis, scrambled_order[:, i_cell], np.newaxis]
                                    use_scrambled = scrambled_activity - np.squeeze(np.repeat(bin_scram_split_mean_preds, n_split_trials_per_scrambled_type, axis=1))
                                else:
                                    use_scrambled = scrambled_activity

                                # use_scrambled = scrambled_minus_mean if key == 'resid_var' else scrambled_activity
                                use_corrs[key]['scram']['split'].dat[start_cell + i_cell, :, :, i_split] = \
                                    find_trial_rew_correls(use_scrambled[np.newaxis, ...], split_scrambled_stat_vec)
                                if key == 'mean':
                                    nolick_scram_inds = [scrambled_inds[i_type, i_cell][
                                                             np.isin(scrambled_inds[i_type, i_cell],
                                                                     np.flatnonzero(nolick_inds))]
                                                         for i_type in range(n_trace_types)]

                                    nolick_scrambled_activity = np.concatenate(
                                        [np.array(use_binned[i_cell, nolick_scram_inds[i_type][i_split::n_splits], :]) for
                                         i_type in range(n_trace_types)], axis=0)

                                    n_split_nolick_trials_per_scrambled_type = [len(nolick_scram_inds[i_type][i_split::n_splits]) for i_type in range(n_trace_types)]

                                    use_corrs['nolick_mean']['scram']['split'].dat[start_cell + i_cell, :, :, i_split] = \
                                        find_trial_rew_correls(nolick_scrambled_activity[np.newaxis, ...], np.concatenate(
                                            [[protocol_info['mean'][i]] * x for i, x in enumerate(n_split_nolick_trials_per_scrambled_type)]))

                                if i_cell == 0:

                                    activity = np.concatenate(
                                        [np.array(use_binned[:, inds[i_type][i_split::n_splits], :]) for i_type in range(n_trace_types)],
                                        axis=1)
                                    # activity_minus_mean = np.concatenate(
                                    #     [np.array(binned_minus_mean[:, inds[i_type][i_split::n_splits], :]) for i_type in
                                    #      range(n_trace_types)], axis=1)
                                    n_trials_per_type = [len(inds[i_type][i_split::n_splits]) for i_type in range(n_trace_types)]

                                    if key in prerew_keys:
                                        stat_vec = np.concatenate(
                                            [[protocol_info[key][i]] * x for i, x in enumerate(n_trials_per_type)])
                                    elif key == 'rew':
                                        # trialwise correlation of actual reward delivered on a given trial, excluding unexpected reward for consistency
                                        stat_vec = np.concatenate(
                                            [rewards[inds[i_type][i_split::n_splits]] for i_type in range(n_trace_types)])
                                    elif key == 'rpe':
                                        # need to repeat this because of the splits
                                        rew_vec = np.concatenate(
                                            [rewards[inds[i_type][i_split::n_splits]] for i_type in
                                             range(n_trace_types)])
                                        # trialwise correlation of RPE (actual minus expected reward)
                                        mean_vec = np.concatenate(
                                            [[protocol_info['mean'][i]] * x for i, x in enumerate(n_trials_per_type)])
                                        stat_vec = rew_vec - mean_vec

                                    if key == 'resid_mean':
                                        # project out the variance regression prediction for all 250 ms bins
                                        split_var_preds = use_corrs['var']['ord']['split'].dat[start_cell:end_cell, np.newaxis, :, 1, i_split] + \
                                                            use_corrs['var']['ord']['split'].dat[start_cell:end_cell, np.newaxis, :, 0, i_split] * \
                                                            np.array(protocol_info['var'])[np.newaxis, :n_trace_types, np.newaxis]
                                        use_activity = activity - np.squeeze(np.repeat(split_var_preds, n_trials_per_type, axis=1))

                                    elif key == 'resid_var' or key == 'resid_cvar':
                                        # project out the value regression prediction for all 250 ms bins
                                        split_value_preds = use_corrs['mean']['ord']['split'].dat[start_cell:end_cell, np.newaxis, :, 1, i_split] + \
                                                            use_corrs['mean']['ord']['split'].dat[start_cell:end_cell, np.newaxis, :, 0, i_split] * \
                                                            np.array(protocol_info['mean'])[np.newaxis, :n_trace_types, np.newaxis]
                                                            # np.array(protocol_info['mean'])[np.newaxis, :n_trace_types, np.newaxis]
                                        use_activity = activity - np.squeeze(np.repeat(split_value_preds, n_trials_per_type, axis=1))
                                    else:
                                        use_activity = activity

                                    # use_activity = activity_minus_mean if key == 'resid_var' else activity
                                    use_corrs[key]['ord']['split'].dat[start_cell:end_cell, :, :, i_split] = \
                                        find_trial_rew_correls(use_activity, stat_vec)
                                    if key == 'mean':
                                        nolick_list = [x[np.isin(x, np.flatnonzero(nolick_inds))] for x in inds]
                                        ntrials_per_nolick_type = [len(x[i_split::n_splits]) for x in nolick_list]
                                        nolick_stat_vec = np.concatenate(
                                            [[protocol_info[key][i]] * x for i, x in enumerate(ntrials_per_nolick_type)])

                                        use_activity = np.concatenate(
                                            [np.array(use_binned[:, nolick_list[i_type][i_split::n_splits], :]) for i_type in
                                             range(n_trace_types)], axis=1)
                                        use_corrs['nolick_mean']['ord']['split'].dat[start_cell:end_cell, :, :, i_split] = \
                                            find_trial_rew_correls(use_activity, nolick_stat_vec)

            start_cell += cells_per_sess[i_ret]


        # in future, I could do something with only those neurons that are modulated during the task,
        # discriminate between odors, or correlate with mean reward, etc.
        odor_comp_ind = 0
        trace_comp_ind = 2
        sub_inds = {}

        # significant modulation during odor period, relative to baseline
        sub_inds['modlu'] = modulation[:, odor_comp_ind] < alpha
        # print(sub_inds['modlu'])

        # ANOVA to look for differences between different trial types during late trace period
        sub_inds['discrim'] = discrim[:, main_comp_ind] < alpha
        # print(sub_inds['discrim'])

        # significant correlation between late trace period activity and mean reward of each trial type
        # I use trialwise correlation (plotted above) instead of tuning curve correlation
        # sub_inds['value_correl'] = sig_value_correls[:, lt_step_ind]
        # print(sub_inds['value_correl'])

        # count number of consecutive significant correlations. If either group is >=min_consec_bins, save it
        my_count = lambda ar: max([sum(1 for _ in group) if key else 0 for key, group in itertools.groupby(ar)])
        min_consec_bins = 3

        def walk(node, prerew_ind_range=None, postrew_ind_range=None, ind_range=None):
            # print('Inside walk')
            for key, item in node.items():
                # print(key)
                if type(item) is dict:
                    if key in prerew_keys:
                        # ind_range = np.arange(odor_on_psth_ind, lt_psth_ind + 1)
                        ind_range = prerew_ind_range
                    elif key in postrew_keys:
                        # ind_range = np.arange(lt_psth_ind + 1, n_psth_bins)
                        ind_range = postrew_ind_range
                    walk(item, prerew_ind_range, postrew_ind_range, ind_range)
                elif type(item) is Strat:
                    # print('Item is Strat!')
                    item.range = ind_range
                    item.sig = item.dat[:, :, 3] < alpha
                    # print(ind_range)
                    # print(item.sig[:, ind_range])
                    consec_bins = np.apply_along_axis(my_count, 1, item.sig[:, ind_range])
                    item.dur = consec_bins * psth_bin_width
                    if key == 'all':
                        item.inds = np.any(item.dat[:, ind_range, 3] < alpha / len(ind_range), axis=1)
                    else:
                        item.inds = np.any(np.any(item.dat[:, ind_range, 3] < alpha / len(ind_range) / 2, axis=1), axis=-1)
                    # print(item.sig)

        walk(corrs, np.arange(odor_on_psth_ind, lt_psth_ind + 1), np.arange(lt_psth_ind + 1, n_psth_bins))
        walk(corrs_seconds, np.arange(odor_ind, n_prerew_periods), np.arange(n_prerew_periods, n_periods))
        # print(corrs_seconds['mean']['ord']['all'].sig)

        # absolute difference between 0 reward and maximum reward are above some threshold
        sub_inds['difference'] = np.abs(X_means[-1, :, main_comp_ind] - X_means[0, :, main_comp_ind]) > min_diff
        # print(sub_inds['difference'])

        # 0 reward and maximum reward are significantly different from one another
        sub_inds['separated'] = separated[:, main_comp_ind] < alpha
        # print(sub_inds['separated'])

        # sub_inds['regress'] = sig_values
        # print(sub_inds['regress'])

        for key, val in zip(neuron_info.keys(), neuron_info.values()):
            if val and type(val[0]) == str:
                neuron_info[key] = np.array(val, dtype='object')
            else:
                neuron_info[key] = np.array(val)

        sub_inds['all'] = np.ones(n_recorded_cells, dtype=bool)

        # what subset to use for analysis. Can change this!!
        all_subset = np.flatnonzero(sub_inds['all'])
        # spec = kwargs['probe1_region'] if table == 'ephys' else kwargs['genotype'].split()[0]
        # spec = kwargs['probe1_region'] if table == 'ephys' else kwargs['genotype'].split()[0]
        if table == 'ephys':
            spec = kwargs['probe1_region']
        elif 'genotype' in kwargs:
            spec = kwargs['genotype'].split()[0]
        else:
            spec = 'combined'
        all_subclass = '_'.join([protocol, table, spec, activity_type, 'all_inds'])  # 'npx_ofc_modlu_inds'  # update this for saving the pop_code fit with the appropriate name
        total_cells = len(all_subset)
        not_subset = np.delete(np.arange(n_recorded_cells), all_subset)
        print('Total cells: ' + str(total_cells))
        # print(corrs['mean']['ord']['all'].__dict__)
        # print(corrs['mean']['scram']['all'].__dict__)

        # value_subset = np.flatnonzero(np.logical_and(sub_inds['value_correl'], diff_resp[:, 0, main_comp_ind] != 0))
        # value_subset = np.flatnonzero(sub_inds['all_value_correl'])
        value_subset = np.flatnonzero(corrs['mean']['ord']['all'].inds)
        not_value = np.delete(np.arange(n_recorded_cells), value_subset)
        n_value = len(value_subset)
        # 'npx_ofc_modlu_inds'  # update this for saving the pop_code fit with the appropriate name
        value_subclass = '_'.join([protocol, table, spec, activity_type, 'all_value_correl_inds'])
        print('Value cells: ' + str(n_value))

        scram_subset = np.flatnonzero(corrs['mean']['scram']['all'].inds)
        # print(scram_subset.shape, scram_subset.max(), n_recorded_cells)
        not_scram = np.delete(np.arange(n_recorded_cells), scram_subset)
        n_scram = len(scram_subset)
        scram_subclass = '_'.join([protocol, table, spec, activity_type, 'all_scram_correl_inds'])
        print('Scrambled cells: ' + str(n_scram))

        subset = all_subset  # for certain analyses, figure out which cells (if any) to select

        n_rpe = np.sum(corrs['rpe']['ord']['all'].inds)
        value_frac = n_value / total_cells
        rpe_frac = n_rpe / total_cells
        n_combined = np.sum(np.logical_and(corrs['mean']['ord']['all'].inds, corrs['rpe']['ord']['all'].inds))

        # n_odor_value = np.sum(sub_inds['odor_correl'])
        # odor_frac = n_odor_value / total_cells
        # n_odor_rpe_combined = np.sum(np.logical_and(sub_inds['odor_correl'], sub_inds['rpe_correl']))

        # pcolormesh won't work if the array is too big, so downsample to 500 time points if necessary
        ds_samps = 500
        if n_samps > ds_samps:
            ds_factor = int(n_samps / ds_samps)
        else:
            ds_factor = 1

        pcolor_time = pcolor_time_full[::ds_factor]
        std_time = (pcolor_time[:-1] + pcolor_time[1:]) / 2  # midpoints of pcolor_time, since pcolor labels both endpoints
        ds_samps = len(std_time)

        peak_inds = np.argmax(high_tt_concat[0], axis=1)
        # sort each array by peak of activity
        sort_peak_inds = np.argsort(peak_inds)

        # cs subtimecourse is identical to timecourses['cs'], except that it uses only a test subset of high_tt trials to
        # compute the average, while sorting based on an independent training set, for use in plotting
        # don't sort/subset/downsample here, so I can subset later!
        cs_subtimecourse = timecourses['cs']['zF'].copy()
        cs_subtimecourse[protocol_info['high_tt']] = high_tt_concat[1]
        # cs_subtimecourse = cs_subtimecourse[:, sort_peak_inds[subset], ::ds_factor]

    if not fast_load:
        # save neuron_info individually for later easy access
        with open(os.path.join(paths['neural_fig_roots'][1], 'pooled', 'preloaded', os.path.basename(save_path)),
                  'wb') as f:
            extended_info = neuron_info.copy()
            extended_info['sub_inds'] = sub_inds
            joblib.dump(extended_info, f)

    local_vars = locals().copy()
    # delete unpickleable types from dictionary
    for key in ['rets', 'ret', 'bret', '_', 'my_count', 'ms_spks', 'mnemonic_mat_ds_inc', 'conn', 'f', 'extended_info',
                'colors', 'walk']:  # query
        if key in local_vars:
            del local_vars[key]

    print(psutil.virtual_memory())
    with open(save_path, 'wb') as f:
        joblib.dump(local_vars, f)
    return local_vars

