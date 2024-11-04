import numpy as np
import numpy.matlib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import SplineTransformer, QuantileTransformer
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib
import matplotlib.pyplot as plt

from scipy import stats, ndimage
import dill
from copy import deepcopy
import cmocean
import time
from datetime import datetime
import json

import os
import sys
sys.path.append('/n/holystore01/LABS/uchida_users/Users/alowet/GLM_Tensorflow_2/code')
import glm_class as glm

sys.path.append('../utils')
from paths import *
from db import select_db, get_db_info, execute_sql, insert_into_db, NumpyEncoder
from matio import loadmat
from protocols import load_params

from streams import extract_data_streams
from basis import *
from glm_utils import *

# import pdb; pdb.set_trace()

def regress_session(name, file_date, file_date_id, meta_time=None, table='ephys', refit=False, lr=5e-3,
                    lambda_start=-1, lambda_end=-4.5, n_lambda=8, mom=.5, reg='group_lasso', se_frac=0.75, l1_ratio=0.9):

    # momentum only used in SGDM optimizer; ignored here because we're using Adam
    # l1_ratio only used in elastic_net
    print('Learning rate', lr, 'Momentum', mom, 'lambda_start', lambda_start, 'lambda_end', lambda_end, 'n_lambda',
          n_lambda, 'regularization', reg, 'se_fraction', se_frac, 'l1_ratio', l1_ratio, 'meta', meta_time, 'table', table)
    lambda_series = 10.0 ** np.linspace(lambda_start, lambda_end, n_lambda)
    print(lambda_series)

    # # Check version and eager execution state
    # print("tensorflow version", tf.__version__)
    # print("keras version", tf.keras.__version__)
    # print("Eager Execution Enabled:", tf.executing_eagerly())
    #
    # # Get the number of replicas
    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of replicas:", strategy.num_replicas_in_sync)
    #
    # # Get available devices
    # devices = tf.config.experimental.get_visible_devices()
    # print("Devices:", devices)
    # print(tf.config.experimental.list_logical_devices('GPU'))

    gpus = tf.config.list_physical_devices('GPU')
    print("GPU Available: ", gpus)
    # print("All Physical Devices", tf.config.list_physical_devices())

    if len(gpus) < 1:
        raise Exception('GPU not found')

    paths = get_db_info()
    timebase, neural, behavior, facemap, raw, ret = extract_data_streams(name, int(file_date), file_date_id, table, meta_time)

    n_cells = neural.mat.shape[0]
    n_trials = neural.mat.shape[1]
    nsamps_per_trial = neural.mat.shape[2]
    dt = neural.dt  # NOTE! This is what sets the GLM bin width, and it's actually being inherited silently from extract_data_streams which isn't ideal
    align_ind = np.argmin(np.abs(timebase))

    print(n_cells, n_trials, nsamps_per_trial, align_ind)

    # Set up "temporal" predictors: cue, previous reward magnitude, previous expectation, and previous RPE, which span the entire trial
    colors, protocol_info, periods, _ = load_params(ret['protocol'])
    trial_types = raw['SessionData']['TrialTypes'][:n_trials] - 1  # need n_trials here for the cut_short case
    rews = raw['SessionData']['RewardDelivered'][:n_trials]
    means = np.array(protocol_info['mean'])[trial_types]
    means[np.isnan(means)] = 0  # unexpected reward -- treat as 0 expectation here

    n_back = 2
    prev_rew_mag = np.zeros((n_back, n_trials, nsamps_per_trial))
    prev_mean = np.zeros((n_back, n_trials, nsamps_per_trial))
    for i in range(1, n_back + 1):
        prev_rew_mag[i - 1] = np.tile(np.insert(rews[:-i], 0, [0.] * i)[:, np.newaxis], (1, nsamps_per_trial))
        prev_mean[i - 1] = np.tile(np.insert(means[:-i], 0, [0.] * i)[:, np.newaxis], (1, nsamps_per_trial))
    prev_rpe = prev_rew_mag - prev_mean

    # stack the temporal functions together
    temporal = np.stack((*prev_rew_mag, *prev_mean), axis=0)
    # ['cs' + str(i) for i in range(protocol_info['n_trace_types'])] + \
    temporal_labels = ['{}-back reward magnitude'.format(i) for i in range(1, n_back + 1)] + \
                      ['{}-back mean'.format(i) for i in range(1, n_back + 1)]

    if 'SameRewSize' not in ret['protocol']:
        temporal = np.stack((*temporal, *prev_rpe), axis=0)
        temporal_labels += ['{}-back RPE'.format(i) for i in range(1, n_back + 1)]

    # temporal = prev_rew_mag.copy()
    # temporal_labels = ['{}-back reward magnitude'.format(i) for i in range(1, n_back + 1)]

    print('Temporal predictors assembled')

    # plt.pcolormesh(temporal[..., 0])
    # _ = plt.yticks(np.arange(.5, temporal.shape[0]), temporal_labels, rotation=45)
    # plt.colorbar()

    n_bases = 7
    bases, basis_widths, basis_centers = create_basis(nsamps_per_trial, n_bases)

    # Temporal predictors are multiplied by a raised cosine basis, to tile the trial
    n_temporal = temporal.shape[0]
    mult_activity = np.full((n_temporal * n_bases + n_bases, n_trials, nsamps_per_trial), np.nan)
    mult_labels = []

    # tile each of the "temporal" predictors with raised cosine bases
    for i in range(n_temporal):
        for j in range(n_bases):
            mult_activity[i * n_bases + j, ...] = temporal[i, ...] * bases[np.newaxis, :, j]
            mult_labels.append(temporal_labels[i] + 'bump{}'.format(j))

    # time in trial get added to mult_activity; it's just the bases themselves
    time_in_trial = np.tile(bases.T[:, np.newaxis, :], (1, n_trials, 1))
    mult_activity[-n_bases:] = time_in_trial
    mult_labels = mult_labels + ['time_in_trialbump{}'.format(i) for i in range(n_bases)]

    # plt.pcolormesh(mult_activity[:, 50, :])
    # _ = plt.yticks(np.arange(.5, mult_activity.shape[0], 5), mult_labels[::5], rotation=45)
    # plt.colorbar()

    filts, pred_filts, filt_time = get_filts(dt)

    # get times, and from that indices, of reward delivery (or expected reward delivery)
    # print(n_trials)
    # print(len(raw['SessionData']['RawEvents']['Trial']))
    rew_times = np.array([raw['SessionData']['RawEvents']['Trial'][i].States.Reward[0] for i in range(n_trials)])
    align_times = np.array([raw['SessionData']['RawEvents']['Trial'][i].States.Foreperiod[-1] for i in range(n_trials)])
    # unexpected reward trials need to have the offset added back in, to match the neural/licking/facemap streams
    rew_times[np.isin(trial_types, protocol_info['exclude_tt'])] += protocol_info['exclude_shift']
    rew_times_aligned = rew_times - align_times
    rew_inds_aligned = np.argmin(np.abs(timebase[:, np.newaxis] - rew_times_aligned[np.newaxis, :]), axis=0)

    # since odor/trace duration is always the same use the mode to figure out where reward is expected. This is superior
    # to using PostRew, because this comes after a delay (the duration of valve opening)
    # exp_rew_times_aligned = np.array([raw['SessionData']['RawEvents']['Trial'][i].States.PostRew[0] for i in range(n_trials)]) - align_times
    exp_rew_times_aligned = rew_times_aligned.copy()
    exp_rew_times_aligned[np.isnan(rew_times_aligned)] = stats.mode(exp_rew_times_aligned, keepdims=False)[0]
    exp_rew_inds_aligned = np.argmin(np.abs(timebase[:, np.newaxis] - exp_rew_times_aligned[np.newaxis, :]), axis=0)

    rew_delivered = rews != 0
    rew_mask = np.arange(nsamps_per_trial) == rew_inds_aligned[:, np.newaxis]
    rew_mask[:, 0] = False  # because nans for no delivery got converted to zero
    exp_rew_mask = np.arange(nsamps_per_trial) == exp_rew_inds_aligned[:, np.newaxis]

    # each row of each array is a single impulse (or zero), which will then get convolved with filter set
    # odor_value = np.zeros((n_trials, nsamps_per_trial))
    # odor_value[:, align_ind] = means

    taus = np.linspace(.1, .9, 5)
    expectiles_per_tt = np.array([[stats.expectile(dist, tau) for tau in taus] for dist in protocol_info['dists']])
    if np.amax(protocol_info['exclude_tt']) == np.amax(trial_types):
        expectiles_per_tt[protocol_info['exclude_tt']] = 0
    expectiles_per_trial = expectiles_per_tt[trial_types]  # shape (n_trials, n_taus)
    expectiles = np.zeros((len(taus), n_trials, nsamps_per_trial))
    expectiles[:, :, align_ind] = expectiles_per_trial.T

    rew_present = np.zeros((n_trials, nsamps_per_trial))
    rew_present[rew_mask] = 1

    rew_mag = np.zeros((n_trials, nsamps_per_trial))
    rew_mag[rew_mask] = rews[rew_delivered]

    rpe = np.zeros((n_trials, nsamps_per_trial))
    rpe[exp_rew_mask] = rews - means

    cue = np.zeros((protocol_info['n_trace_types'], n_trials, nsamps_per_trial))
    for i in range(protocol_info['n_trace_types']):
        cue[i, trial_types == i, align_ind] = 1

    # get db_info and preallocate fit_sqls, so that it can be filled up incrementally as we loop through to_drops
    with open(os.path.join(paths['config'])) as json_file:
        db_info = json.load(json_file)
    fit_keys = [x[0] for x in db_info['glm_fit_fields']]  # keys in the db
    # print(fit_keys)

    # fit_sqls = [np.empty((n_cells, len(fit_keys)), dtype='object') for _ in range(len(se_fracs))]  # back when there were multiple se_fracs
    fit_sql = np.empty((n_cells, len(fit_keys)), dtype='object')

    # start populating array containing information for glm_fit, which combines full model, drop_expectiles, etc. in a single
    # entry, but importantly disaggregates by se_frac, regularization, and learning rate (where applicable)
    # fit_sql = fit_sqls[i_frac]
    for i_key, key in enumerate(fit_keys):
        if key in ret.keys():
            fit_sql[:, i_key] = ret[key]
    fit_sql[:, fit_keys.index('i_cell')] = np.arange(n_cells)
    fit_sql[:, fit_keys.index('se_frac')] = se_frac
    fit_sql[:, fit_keys.index('regularization')] = reg
    fit_sql[:, fit_keys.index('learning_rate')] = lr
    fit_sql[:, fit_keys.index('l1_ratio')] = l1_ratio
    fit_sql[:, fit_keys.index('dt')] = dt
    fit_sql[:, fit_keys.index('modality')] = table

    # stack the conv functions together
    # rew present and reward magnitude are *identical* for Bernoulli, which is bad. So omit the "reward present"
    # regressor in that case
    to_drops = ['none', 'licking', 'expectiles', 'motor', 'history', 'reward']

    for i_drop, to_drop in enumerate(to_drops):

        print('drop', to_drop)

        # look for results file
        figure_path = os.path.join(paths['neural_fig_roots'][0], name, file_date_id)
        # pickle_path = os.path.join(figure_path, 'regress_cells_drop_{}_lr_{}.p'.format(to_drop, lr))
        if reg == 'elastic_net':
            pickle_path = os.path.join(figure_path, 'regress_cells_drop_{}_lr_{}_reg_{}_l1_ratio_{}.p'.format(  # _dt_{}
                to_drop, lr, reg, l1_ratio))  # l1_ratio will be irrelevant if reg == 'group_lasso', but include for consistency
        else:
            pickle_path = os.path.join(figure_path, 'regress_cells_drop_{}_lr_{}_reg_{}.p'.format(to_drop, lr, reg))  #_dt_{}


        all_in_db = select_db('my', 'glm_setup', '*', 'name=? AND exp_date=? AND learning_rate=? AND lambda_series=? ' + \
                              'AND regularization=? AND se_frac=? AND l1_ratio=?',  # AND dt=?
                              (name, file_date, lr, json.dumps(lambda_series, cls=NumpyEncoder), reg, se_frac, l1_ratio), unique=False)
        dropped_in_db = select_db('my', 'glm_setup', '*', 'name=? AND exp_date=? AND learning_rate=? AND ' + \
                                  'lambda_series=? AND regularization=? AND se_frac=? AND l1_ratio=? AND dropped_out_vars=?',  # AND dt=?
                                 (name, file_date, lr, json.dumps(lambda_series, cls=NumpyEncoder), reg, se_frac, l1_ratio, to_drop), unique=False)
        print('all_in_db len', len(all_in_db), 'dropped_in_db len', len(dropped_in_db))

        # print(len(dropped_in_db))
        # print(pickle_path)
        # print(os.path.isfile(pickle_path))
        # print(not refit)

        if len(dropped_in_db) > 0 and os.path.isfile(pickle_path) and not refit:  # this can happen e.g. if the job gets requeued
        # if os.path.isfile(pickle_path) and not refit:  # this can happen e.g. if the job gets requeued
            # print('Session already fit {} with learning rate {}. Skipping.'.format(to_drop, lr))
            # continue
            # LOAD MODEL FROM DISK
            # load regressor matrix. Doesn't depend on learning_rate, regularization, l1_ratio, or se_fraction
            with open(os.path.join(figure_path, 'regress_drop_{}.p'.format(to_drop)), 'rb') as f:  # _dt_{}
                dat = pickle.load(f)
            print('Found ' + pickle_path)
            with open(pickle_path, 'rb') as f:
                model = dill.load(f)
            # bring loaded variables into namespace
            # for var in container.keys():
            #     exec("{} = container['{}']".format(var, var))

        else:
            if 'reward' not in to_drop:
                if 'Bernoulli' in ret['protocol']:
                    conv = [rew_mag, rpe]
                    conv_base_labels = ['Reward magnitude', 'RPE']
                elif 'SameRewSize' in ret['protocol']:
                    print('Omit RPE')
                    conv = [rew_present, rew_mag]
                    conv_base_labels = ['Reward present', 'Reward magnitude']
                else:
                    conv = [rew_present, rew_mag, rpe]
                    conv_base_labels = ['Reward present', 'Reward magnitude', 'RPE']
            else:
                conv = []
                conv_base_labels = []

            # if 'value' not in to_drop:
            #     conv.append(odor_value)
            #     conv_base_labels.append('Value')
            if 'expectiles' not in to_drop:
                if 'SameRewSize' in ret['protocol']:
                    # All the expectiles predict the same thing for all trial types, so this reduces to just value
                    # Include only the 0.5th expectile in this case
                    print(ret['protocol'])
                    conv.extend([*expectiles[taus == 0.5]])
                    conv_base_labels.append('{:.1f}-expectile'.format(0.5))
                else:
                    conv.extend([*expectiles])
                    conv_base_labels.extend(['{:.1f}-expectile'.format(tau) for tau in taus])
            if 'licking' not in to_drop and 'motor' not in to_drop:
                conv.append(behavior.mat)
                conv_base_labels.append('Licking')
            if 'none' in to_drop or 'motor' in to_drop:
                conv.extend([*cue])
                conv_base_labels.extend(protocol_info['trace_type_names'])
            if 'motor' not in to_drop:
                # [print(mat.shape) for mat in conv]
                # [print(mat.shape) for mat in facemap.conv]
                conv = np.stack((*conv, *facemap.conv), axis=0)
                conv_base_labels.extend([x for x in ['whisking', 'running'] if x in facemap.dat['dat'].keys()])

            conv = np.array(conv)
            print(conv.shape)
            print('Conv predictors assembled. Convolving...')

            n_conv = conv.shape[0]
            n_filts = filts.shape[1]
            double_conv = ['Licking', 'whisking', 'running']  # predictive as well as responsive signals
            conv_activity = np.full((n_conv * n_filts, n_trials, nsamps_per_trial), np.nan)
            conv_labels = []

            # convolve each of the "conv" predictors with log-scaled raised cosine bases (responsive bases)
            for i in range(n_conv):
                for j in range(n_filts):
                    conv_activity[i * n_filts + j, :, :] = ndimage.convolve1d(conv[i, :, :], filts[:, j], axis=1, mode='nearest')
                    conv_labels.append(conv_base_labels[i] + 'filt{}'.format(j))

            # furthermore, convolve the predictive bases
            for double in double_conv:
                if double in conv_base_labels:
                    i_conv = conv_base_labels.index(double)
                    pred_activity = np.full((n_filts, n_trials, nsamps_per_trial), np.nan)
                    for j in range(n_filts):
                        pred_activity[j] = ndimage.convolve1d(conv[i_conv, :, :], pred_filts[:, j], axis=1, mode='nearest')
                        conv_labels.append(conv_base_labels[i_conv] + '_predfilt{}'.format(j))
                    conv_activity = np.concatenate((conv_activity, pred_activity), axis=0)

            print('Convolution complete.')

            # All regressors (including facemap data) are stacked together and then reshaped into a giant regressor matrix
            if 'motor' in to_drop:
                regressor_3d = np.stack(
                    (*mult_activity,
                     *conv_activity), axis=0
                )
                regressor_labels = mult_labels + conv_labels
            elif 'history' in to_drop:
                regressor_3d = np.stack(
                    (*mult_activity[-n_bases:],
                     *conv_activity,
                     *facemap.nonconv), axis=0
                )
                regressor_labels = mult_labels[-n_bases:] + conv_labels + facemap.labels
                # print(regressor_3d.shape, len(regressor_labels), regressor_labels)
            else:
                regressor_3d = np.stack(
                    (*mult_activity,
                     *conv_activity,
                     *facemap.nonconv), axis=0
                )
                regressor_labels = mult_labels + conv_labels + facemap.labels

            # reshape matrix so that it is 2d: n_predictors x n_time_points
            regressor_mat = regressor_3d.reshape(regressor_3d.shape[0], n_trials * nsamps_per_trial)
            time_in_trial_all = np.tile(timebase, n_trials)

            # For mult regressors, the end of excluded_trials should be trimmed off.
            # But for conv and facemap regressors, as well as neural data, the beginning should be trimmed off,
            # because it's aligned to reward delivery
            excluded_trial_types = np.isin(trial_types, protocol_info['exclude_tt'])
            excluded_trials = np.flatnonzero(excluded_trial_types)

            print(len(excluded_trials))
            if len(excluded_trials) > 0:

                n_inds_to_trim = int(round((neural.dat['timestamps']['stim'] + neural.dat['timestamps']['trace']) / dt))
                mult_excluded_per_trial = np.arange(nsamps_per_trial - n_inds_to_trim, nsamps_per_trial)
                conv_excluded_per_trial = np.arange(n_inds_to_trim)

                mult_excluded_samples = np.concatenate(
                    np.array([x * nsamps_per_trial + mult_excluded_per_trial for x in excluded_trials]))
                conv_excluded_samples = np.concatenate(
                    np.array([x * nsamps_per_trial + conv_excluded_per_trial for x in excluded_trials]))

                mult_trim_mat = np.delete(regressor_mat[:len(mult_labels)], mult_excluded_samples, axis=1)
                conv_trim_mat = np.delete(regressor_mat[len(mult_labels):], conv_excluded_samples, axis=1)
                time_in_trial = np.delete(time_in_trial_all, conv_excluded_samples)

            else:
                mult_trim_mat = regressor_mat[:len(mult_labels)]
                conv_trim_mat = regressor_mat[len(mult_labels):]
                time_in_trial = time_in_trial_all

            # Nuissance regressors are added to account for drift
            n_samples = conv_trim_mat.shape[1]
            n_nuissance = 20
            nuissance_bases, nuissance_widths, nuissance_centers = create_basis(n_samples, n_nuissance)

            regressor_mat = np.concatenate((mult_trim_mat, conv_trim_mat, nuissance_bases.T), axis=0)
            regressor_labels = regressor_labels + ['nuissancebump{}'.format(i) for i in range(n_nuissance)]
            n_regressors = regressor_mat.shape[0]

            # regressor_mat is shape (n_regressors, n_timepoints)
            # z-score
            z_mat = stats.zscore(regressor_mat, axis=1)
            X = z_mat.T
            print(len(regressor_labels))
            print(X.shape)
            print('Regressor matrix created')

            # plot regressor matrix
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.pcolormesh(np.arange(0, nsamps_per_trial * 20 + 1, 10), np.arange(n_regressors + 1),
                          z_mat[:, :nsamps_per_trial * 20:10])
            ax.set_ylim([n_regressors, 0])
            ax.set_yticks(np.arange(0, n_regressors, 2))
            ax.set_yticklabels(regressor_labels[::2], rotation=45)

            fig.savefig(os.path.join(figure_path, 'regressor_mat_drop_{}.png'.format(to_drop)), dpi=300,
                        bbox_inches='tight')

            # plot correlations between each row of regressor matrix
            cmap = cmocean.tools.crop(cmocean.cm.balance, vmin=-.7, vmax=1, pivot=0)
            fig, ax = plt.subplots(figsize=(20, 16))
            corrs = np.corrcoef(z_mat)
            # np.fill_diagonal(corrs, np.nan)
            plt.pcolormesh(np.arange(n_regressors + 1), np.arange(n_regressors + 1), corrs, cmap=cmap)
            plt.ylim([n_regressors, 0])
            plt.yticks(np.arange(0, n_regressors, 5), regressor_labels[::5], rotation=45)
            plt.xticks(np.arange(0, n_regressors, 5), regressor_labels[::5], rotation=90)
            plt.colorbar()
            fig.savefig(os.path.join(figure_path, 'regressor_mat_corrs_drop_{}.png'.format(to_drop)), dpi=300,
                        bbox_inches='tight')
            plt.show()

            # get neural data
            yall = neural.mat.reshape(n_cells, n_trials * nsamps_per_trial)

            # Split into train and test sets at trial level
            # vector giving the trial number of each sample
            trial_num = np.tile(np.arange(n_trials)[:, np.newaxis], (1, nsamps_per_trial)).reshape(-1, )

            if len(excluded_trials) > 0:
                # delete nans corresponding to X
                Y = np.delete(yall, conv_excluded_samples, axis=1).T
                trial_num = np.delete(trial_num, conv_excluded_samples)
            else:
                Y = yall.T

            # for AL17/20210331, the first trial neural response was all nans for all neurons. Remove these
            remaining_nans = np.isnan(Y[:, 0])
            Y = np.delete(Y, remaining_nans, axis=0)
            X = np.delete(X, remaining_nans, axis=0)
            trial_num = np.delete(trial_num, remaining_nans)

            # Get indices for splitting according to trial_id
            n_samples = X.shape[0]
            seed = ret['mid'] * 100 + ret['sid']
            gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=seed)
            train_idx, test_idx = next(gss.split(X, Y, trial_num))
            print(test_idx)  # note that the train/test splits are identical because we set the random state

            # Split data into train and test set
            X_train = X[train_idx, :]
            y_train = Y[train_idx, :]
            X_test = X[test_idx, :]
            y_test = Y[test_idx, :]
            trial_id_train = trial_num[
                train_idx]  # extract trial_id for training data, which is used in CV splits later during fitting

            group_size, group_name, group_ind = parse_group_from_feature_names(regressor_labels)

            dat = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'labels': regressor_labels,
                'bases': bases,
                'n_back': n_back,
                'nsamps_per_trial': nsamps_per_trial,
                'filts': filts,
                'filt_time': filt_time,
                'conv_base_labels': conv_base_labels,
                'trial_num': trial_num,
                'time_in_trial': time_in_trial,
                # 'train_frames': train_frames,
                'n_cells': n_cells,
                'n_types': protocol_info['n_trace_types'],
                'trial_id_train': trial_id_train,
                'group_size': group_size,
                'group_name': group_name,
                'group_ind': group_ind
            }
            print('Saving regressor mat')
            with open(os.path.join(figure_path, 'regress_drop_{}.p'.format(to_drop)), 'wb') as f:
                pickle.dump(dat, f)

            plt.close('all')

            print('fitting')
            print(dat['X_train'].shape)

            # Reset keras states
            tf.keras.backend.clear_session()

            # for lr in [5e-3]:  # 5e-3, 1e-2, 5e-2, 1e-1]:
            start_time = time.time()

            # Initialize GLM_CV (here we're only specifying key input arguments; others are left as default values; see documentation for details)
            model = glm.GLM_CV(n_folds=5, auto_split=True, split_by_group=True,
                               activation='exp', loss_type='poisson', momentum=mom,
                               regularization=reg, lambda_series=lambda_series,
                               optimizer='adam', learning_rate=lr, l1_ratio=l1_ratio)

            # Fit the GLM_CV on training data
            try:
                model.fit(dat['X_train'], dat['y_train'], group_idx=dat['trial_id_train'], feature_group_size=dat['group_size'],
                          verbose=True)
                fitting_time = time.time() - start_time

                # add to database
                all_keys = [key[0] for key in db_info['glm_setup_fields']]  # keys in the db
                db_dict = {'i_model': i_drop, 'fit_date': datetime.today().strftime('%Y%m%d'),
                           'n_cells': dat['n_cells'], 'fitting_time': fitting_time, 'dropped_out_vars': to_drop,
                           'regressor_labels': json.dumps(regressor_labels), 'se_frac': se_frac, 'seed': seed, 'modality': table, 'dt': dt}
                for key in all_keys:
                    if key in ret.keys():
                        db_dict[key] = ret[key]
                    elif key == 'lambda_series':
                        db_dict[key] = json.dumps(getattr(model, key), cls=NumpyEncoder)
                    elif key in model.__dict__.keys():
                        db_dict[key] = getattr(model, key)

                # print(db_dict)
                # print(dt)
                # print(db_dict['dt'])
                insert_into_db('my', 'glm_setup', tuple(db_dict.keys()), tuple(db_dict.values()))

                chkdir(pickle_path)
                with open(pickle_path, 'wb') as f:
                    dill.dump(model, f)

            except AssertionError:
                print('Loss was nan. This may be due to a learning rate that is too high, but could also just be ' + \
                      'because the problem is ill-conditioned in some way. Skip for now')

        if model.fitted:

            # Select models based on CV performance
            model.select_model(se_fraction=se_frac, make_fig=True)

            # Evaluate model performance on test data
            # print(np.sum(np.isnan(dat['X_test'])), np.sum(np.isnan(dat['y_test'])))
            # print(model.__dict__)
            frac_dev_expl, dev_model, dev_null, dev_expl = model.evaluate(dat['X_test'], dat['y_test'], make_fig=True)

            # Evaluate the model after zeroing out nuissance regressors
            ablate_ind = [this_ind in [dat['group_name'].index('nuissance')] for this_ind in dat['group_ind']]
            X_ablated = dat['X_test'].copy()
            X_ablated[:, ablate_ind] = 0
            frac_dev_expl_abl, dev_model_abl, dev_null_abl, dev_expl_abl = model.evaluate(X_ablated, dat['y_test'])

            if i_drop == 0:
                fit_sql[:, fit_keys.index('lambda')] = model.selected_lambda
                fit_sql[:, fit_keys.index('lambda_ind')] = model.selected_lambda_ind
                fit_sql[:, fit_keys.index('null_dev')] = dev_null

            prefix = 'full' if to_drop == 'none' else to_drop
            for suffix, fitval in zip(['coefs', 'dev', 'dev_expl', 'dev_abl_nuissance', 'dev_expl_abl_nuissance'],
                                      [model.selected_w, dev_model, frac_dev_expl, dev_model_abl, frac_dev_expl_abl]):
                if suffix == 'coefs': fitval = [json.dumps(coefs, cls=NumpyEncoder) for coefs in fitval.T]
                fit_sql[:, fit_keys.index('_'.join([prefix, suffix]))] = fitval

    # print(fit_keys)
    # print(fit_sql[0])
    insert_into_db('my', 'glm_fit', fit_keys, [tuple(entry) for entry in fit_sql], many=True)
    print('Inserted fits into glm_fit table.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Regress cells')
    parser.add_argument('name')
    parser.add_argument('file_date', type=int)
    parser.add_argument('file_date_id')
    parser.add_argument('meta_time', type=int, default=None)
    parser.add_argument('-t', '--table', default='ephys')
    parser.add_argument('-r', '--refit', type=int, default=0)
    parser.add_argument('-l', '--lr', type=float, default=5e-3)
    parser.add_argument('-s', '--lambda_start', type=float, default=-1)
    parser.add_argument('-e', '--lambda_end', type=float, default=-4.5)
    parser.add_argument('-n', '--n_lambda', type=int, default=8)
    parser.add_argument('-o', '--momentum', type=float, default=.5)
    parser.add_argument('-g', '--regularization', default='group_lasso')  # or 'elastic_net'
    parser.add_argument('-f', '--se_frac', type=float, default=.75)
    parser.add_argument('-i', '--l1_ratio', type=float, default=.9)

    args = parser.parse_args()
    regress_session(args.name, args.file_date, args.file_date_id, args.meta_time, args.table, args.refit, args.lr,
                    args.lambda_start, args.lambda_end, args.n_lambda, args.momentum, args.regularization, args.se_frac, args.l1_ratio)
