from scipy import signal, stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from scipy import io
import os
import glob
import datetime
import sqlite3
from sqlite3 import Error

import multiprocessing as mp
import socket
from cycler import cycler
import json
import sys
import pandas as pd

sys.path.append('../utils')
from matio import loadmat
from paths import raise_print
from protocols import load_params


def check_stats(rets, stat, pairs_to_check, alpha=0.05):
    if stat != 'NULL':
        clean_rets = []
        for ret in rets:
            ret = {key: ret[key] for key in ret.keys()}
            table_stats = json.loads(ret['stats'])
            if stat in ['p_anova', 'p_kruskal'] and table_stats[stat] < alpha:
                clean_rets.append(ret)
            elif stat == 'tukeyHSD':
                df = pd.DataFrame(table_stats['tukeyHSD'][1:], columns=table_stats['tukeyHSD'][0])
                add_ret = True
                for index, row in df.iterrows():
                    if (row['group1'], row['group2']) in pairs_to_check and \
                            (row['p-adj'] >= alpha or row['meandiff'] < 1):
                        add_ret = False
                if add_ret:
                    clean_rets.append(ret)
        return clean_rets
    else:
        return rets


def getEventTraces(protocol, sr, data, trial_list, trial_types, time, data_type, plot_auxfigs):
    """
    This function is to be called by the Plot_SingleSession function.
    """
    standard_protocols = ['DistributionalRL_6Odours', 'Bernoulli', 'Shock6Odor', 'ShockBernoulli', 'SameRewDist',
                          'StimGradient', 'TestStim', 'SameRewSize', 'SameRewVar']
    if protocol in standard_protocols:
        # stim_state = 'Stimulus' + str(trial_types) + 'Delivery'
        trial_start_state_name = 'Foreperiod'
    else:
        raise_print('protocol not recognized')

    _, protocol_info, _, _ = load_params(protocol)

    if sr == 1000:  # This is the SR I will use in general
        decround = 3
    elif sr == 10000:
        decround = 4  # This is the precision with which the data was acquired (0.1 ms)
    else:
        raise_print('sampling rate not recognized')

    if data_type == 'Lick':
        port1 = 'Port1In'
        data = data['RawEvents']
        smooth_mult = 1000
        smooth_win = signal.windows.gaussian(smooth_mult, .05 * smooth_mult)  # number of points in window, SD of window
        smooth_win = smooth_win / np.sum(smooth_win)
    else:
        raise_print('data types other than "Lick" not currently supported')

    # data_toplot = np.full((np.size(trial_list),np.size(time)), np.nan)
    n_trials = np.size(trial_list)
    data_toplot = np.zeros((n_trials, np.size(time)))
    cs_in = np.zeros((n_trials,))
    cs_out = np.zeros((n_trials,))
    iti_start = np.zeros((n_trials,))
    trace_start = np.zeros((n_trials,))
    trace_end = np.zeros((n_trials,))

    # Loop across trials of this type (or all types, depending on what trial_list is)
    for i, i_trial in enumerate(trial_list):

        if protocol in standard_protocols:
            stim_state = 'Stimulus' + str(trial_types[i]) + 'Delivery'

        # align to the first sample after the end of the foreperiod
        align_time = float(data['Trial'][i_trial].States.Foreperiod[-1])
        start_state = getattr(data['Trial'][i_trial].States, trial_start_state_name)
        trial_start = start_state[0] - align_time
        trial_start_ind = np.flatnonzero(np.round(time, decround) == np.round(trial_start, decround))[0]

        # trial ends at last sample of ITI
        trial_end = data['Trial'][i_trial].States.ITI[-1] - align_time
        trial_end_ind = np.flatnonzero(np.round(time, decround) == np.round(trial_end, decround))[0]
        if np.size(trial_start_ind) == 0: trial_start_ind = 0
        if np.size(trial_end_ind) == 0: trial_end_pos = -1

        iti_start[i] = data['Trial'][i_trial].States.ITI[0] - align_time
        if hasattr(data['Trial'][i_trial].States, stim_state):
            state_times = getattr(data['Trial'][i_trial].States, stim_state)
            cs_in[i] = state_times[0] - align_time
            cs_out[i] = state_times[1] - align_time
            trace_start[i] = data['Trial'][i_trial].States.Trace[0] - align_time
            trace_end[i] = data['Trial'][i_trial].States.Trace[1] - align_time
        # else:
        #     cs_in[i] = iti_start[i]
        #     cs_out[i] = iti_start[i]

        if data_type == 'Lick':
            # Look for the event of interest in this period
            if hasattr(data['Trial'][i_trial].Events, port1):
                if hasattr(data['Trial'][i_trial].States, stim_state):
                    lick_in = getattr(data['Trial'][i_trial].Events, port1) - align_time
                else:  # unexpected reward
                    lick_in = getattr(data['Trial'][i_trial].Events, port1) - align_time + protocol_info['exclude_shift']
                # lick_out = getattr(data['Trial'][i_trial].Events, port1out) - align_time
                # ugly exception if there is only one lick
                if type(lick_in) is not np.ndarray:
                    lick_in = np.array([lick_in])
                lick_in_ind = np.zeros((np.size(lick_in)), dtype='int16')
                for j in range(np.size(lick_in)):
                    if lick_in[j] < time[-1]:
                        lick_in_ind[j] = np.flatnonzero(np.round(time, decround) == np.round(lick_in[j], decround))[0]
                        data_toplot[i, lick_in_ind[j]] = 1
        else:
            raise_print('Events other than licks not currently supported')

    data_toplot_smoothed = np.zeros(np.shape(data_toplot))
    for k in range(np.size(data_toplot, 0)):
        data_toplot_smoothed[k, :] = smooth_mult * np.convolve(data_toplot[k, :], smooth_win, 'same')
    if plot_auxfigs:
        """
        plt.figure()
        plt.pcolor(time, np.arange(np.size(data_toplot,0)), data_toplot)
        for i in range(np.size(data_toplot,0)):
            plt.plot([cs_in[i], cs_in[i]], [i, i+1], color=(0,0,0))
            plt.plot([cs_out[i], cs_out[i]], [i, i+1], color=(0,0,0))
            plt.plot([iti_start[i], iti_start[i]], [i, i+1], color=(0,0,0))
        plt.show()
        """
        plt.figure()
        plt.pcolor(time, np.arange(np.size(data_toplot_smoothed, 0)), data_toplot_smoothed)
        for i in range(np.size(data_toplot, 0)):
            # could all just be plt.axvline
            plt.plot([cs_in[i], cs_in[i]], [i, i + 1], color=(0, 0, 0))
            plt.plot([cs_out[i], cs_out[i]], [i, i + 1], color=(0, 0, 0))
            plt.plot([iti_start[i], iti_start[i]], [i, i + 1], color=(0, 0, 0))
        plt.show()

    return data_toplot_smoothed, data_toplot, cs_in, cs_out, trace_start, trace_end

def zero_runs(a):
    # https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
#     print(absdiff)
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
#     print(ranges)
    diffranges = ranges[:, 1] - ranges[:, 0]
    return ranges, diffranges


def compute_lme_pval(mfit):
    # the above doesn't give enough sig figs, so copied the actual p-value computation from statsmodels source
    sdf = np.nan * np.ones((mfit.k_fe + mfit.k_re2 + mfit.k_vc, 6))
    # Coefficient estimates
    sdf[0:mfit.k_fe, 0] = mfit.fe_params
    # Standard errors
    sdf[0:mfit.k_fe, 1] = np.sqrt(np.diag(mfit.cov_params()[0:mfit.k_fe]))
    # Z-scores
    sdf[0:mfit.k_fe, 2] = sdf[0:mfit.k_fe, 0] / sdf[0:mfit.k_fe, 1]
    # p-values
    pval = (2 * stats.norm.cdf(-np.abs(sdf[0:mfit.k_fe, 2])))
    return pval

def setUpLickingTrace(trace_dict, ax=None, override_ylims=False):
    if ax is not None:
        plt.sca(ax)

    plt.axvspan(np.mean(trace_dict['cs_in']), np.mean(trace_dict['cs_out']), alpha=.5, facecolor=(.8, .8, .8),
                label='_nolegend_')
    plt.axvline(np.mean(trace_dict['trace_end']), alpha=.5, color=(.8, .8, .8), label='_nolegend_')
    plt.ylabel(trace_dict['ylabel'])
    if not override_ylims:
        plt.ylim(0, 15)
    plt.xlim(trace_dict['xlim'])
    plt.xlabel(trace_dict['xlabel'])


def plotLickingTrace(trace_dict, mean_lick_pat, sem_lick_pat, ax=None):
    # plt.axvspan(np.mean(trace_dict['cs_in']), np.mean(trace_dict['cs_out']), alpha=.5, color=(.8,.8,.8))
    # plt.axvline(np.mean(trace_dict['trace_end']), alpha=.5, color=(.8,.8,.8))

    # if len(mean_lick_pat.shape) == 1:

    if ax is not None:
        plt.sca(ax)

    mean_lick_pat_slice = mean_lick_pat[trace_dict['pos1_time']:trace_dict['pos2_time'] + 1]
    sem_lick_pat_slice = sem_lick_pat[trace_dict['pos1_time']:trace_dict['pos2_time'] + 1]

    # time = np.hstack((trace_dict['time_toplot'], np.flip(trace_dict['time_toplot'], axis=0)))

    # plt.fill(time, np.hstack((mean_lick_pat_slice + sem_lick_pat_slice, \
    # 						  np.flip(mean_lick_pat_slice - sem_lick_pat_slice, axis=0))), \
    # 		 color=trace_dict['colors'], ec=trace_dict['colors'], alpha=0.2)

    plt.fill_between(trace_dict['time_toplot'], mean_lick_pat_slice + sem_lick_pat_slice, mean_lick_pat_slice -
                     sem_lick_pat_slice, color=trace_dict['colors'], edgecolor=trace_dict['colors'], alpha=0.2)

    handle = plt.plot(trace_dict['time_toplot'], mean_lick_pat_slice, color=trace_dict['colors'], lw=1)
    return handle


def get_beh_session(beh_data_folder, day):
    session = glob.glob(beh_data_folder + '/*' + day + '*.mat')
    if len(session) == 0:
        raise_print('Could not find file from ' + day + ' in ' + beh_data_folder)
    else:
        data = []
        for datafile_path in session:
            datafile_name = os.path.basename(os.path.normpath(datafile_path))
            converted_data = loadmat(datafile_path)
            session_datum = converted_data['SessionData']
            session_time = datafile_name.split('_')[-1].replace('.mat', '')
            data.append((datafile_name, session_datum, session_time))
    return data


def get_alphas(taus, avg):
    """
    For a given alpha_avg, compute a set of alpha plus's and alpha minus's
    """
    alpha_ps = taus * avg
    alpha_ns = avg - alpha_ps
    return alpha_ps, alpha_ns


def plot_v(n_chan, rew, method, row, col, gs, n_trials, alpha_avg, N_CHAN, taus, colors, bin_midpoints,
           bounds, grey, stretch, bin_edges):

    # tweak learning rate
    if method == 'expec':
        alpha_avg = alpha_avg / 4

    # get alphas
    if n_chan == N_CHAN:
        if taus is None:
            taus = np.linspace(0.1, 0.9, N_CHAN)
        alpha_ps, alpha_ns = get_alphas(taus, alpha_avg)
    else:
        alpha_ps, alpha_ns = np.array(alpha_avg), np.array(alpha_avg)

    # initialize value predictor(s)
    assert n_chan == len(taus)
    V_i = 2 * np.ones((n_trials, n_chan))
    for iTrial in range(n_trials - 1):
        # compute delta (or delta_i's) as r - V
        delta = rew[iTrial] - V_i[iTrial, :]
        indic = delta <= 0
        if method == 'expec':
            #  Delta_V = alpha * delta, converges to mean/expectiles
            update = delta * (indic * alpha_ns + (1 - indic) * alpha_ps)
        elif method == 'quant':
            #  Delta_V = alpha * sign(delta), converges to median/quantiles
            update = np.sign(delta) * (indic * alpha_ns + (1 - indic) * alpha_ps)
        # update value predictors accordingly
        V_i[iTrial + 1, :] = V_i[iTrial, :] + update

    # plot on appropriate axes
    ax = plt.subplot(gs[row, col])

    # set colormap
    if n_chan == 1:
        ax.set_prop_cycle(color=[colors[N_CHAN // 2]])
    else:
        ax.set_prop_cycle(color=colors)

    ax.plot(np.arange(n_trials), V_i)
    ax.set_xlabel('Trial')
    # if col == 0:
    #     ax.set_ylabel('Activity')
    # else:
    #     ax.set_yticklabels([])
    ax.set_ylim(bounds)
    ax.set_xticks([0, n_trials])
    ax.set_xlim([0, n_trials * stretch])

    # plot quantile/expectile lines
    if n_chan == N_CHAN:
        lines = ax.hlines(y=V_i[-1, :], xmin=n_trials * 1.1, xmax=n_trials * 1.19, lw=1, colors=colors)
        lines.set_clip_on(False)

    # # plot reward histogram
    # hist, _ = np.histogram(rew, bin_edges, density=True)
    # rew_ax = plt.subplot(gs[row, col + 1])
    # rew_ax.barh(bin_midpoints, hist * np.mean(np.diff(bin_edges)), color=grey)  # multiply it so it's a PMF not a PDF
    # rew_ax.set_xlim([0, 1])
    # rew_ax.set_xlabel('Prob')
    # rew_ax.set_yticklabels([])
    #     plot_rew(rew_ax, bin_midpoints, hist, grey, bounds)
    return ax


def get_colors(n_colors, cmap, vmin=0, vmax=1):
    """
    For a given colormap, return a list of colors spanning that colormap.
    """
    return cmap(np.linspace(vmin, vmax, n_colors))
