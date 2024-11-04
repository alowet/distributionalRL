import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from traceUtils import *
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import sys
import os
from operator import itemgetter
import pickle
import json
import subprocess
from datetime import datetime
import glob
import sys

sys.path.append('../utils')
from plotting import *
from paths import *
from db import *
from protocols import *


def plotSingleSession(mouse, protocol, day, savefiles=1, has_opto=0):
    """
    This function plots licking data from a single session.
    Translated from Matlab to Python by Adam S. Lowet, Oct. 8-13, 2019
    Created by Sara Matias
    e.g. mouse: 'D1-01'
    e.g. protocol: 'DistributionalRL6Odours'
    e.g. day: '20191007'
    e.g. savefiles: 0 or 1, True or False

    plotSingleSession('D1-01', 'DistributionalRL6Odours', '20191007', 1)
    """

    ### General Parameters ###
    plt.close('all')
    plot_auxfigs = 0
    sr = 1000  # sampling rate to use (though acquired with 10000 Hz precision)
    colors, vline_color, trial_type_names, trace_type_names, n_trial_types, n_trace_types, variable_rew_css = get_cs_info(
        protocol)
    _, protocol_info, _, _ = load_params(protocol)  # some redundancy here, but ah well
    db = get_db_info()

    data_root = db['behavior_root']
    plot_roots = db['behavior_fig_roots']

    rew_code = {2: 0, 4: 1, 6: 2}  # key = reward amount in uL, value = aux value for plotting
    #   Trial types: TT1 no reward; TT2: 80% 4ul and 10% 2 and 6 ul; TT3: 50% 2ul + 50% 6ul,
    #   TT4: uniform distribution 1-7ul (mean 4ul); TT5: 100% 2ul; TT6: 100% 6ul.

    # Database location and settings:
    alpha = 0.05  # for significance testing on one-way ANOVA

    ### Plotting settings ###
    lg_font_size = 12
    sm_font_size = 10  # might as well make it all pretty and such while we're at it
    params = {'font.weight': 'normal',
              'font.size': 8,
              'text.usetex': False,  # disabling external LaTex only. Makes things WAY easier!!
              'mathtext.fontset': 'stixsans',
              'axes.titlesize': lg_font_size,
              'axes.labelsize': lg_font_size,
              'legend.fontsize': sm_font_size,
              'xtick.labelsize': sm_font_size,
              'ytick.labelsize': sm_font_size
              }
    matplotlib.rcParams.update(params)

    ### Load the data ###

    # copy over from holylfs to scratch, because the FAS RC cluster is ridiculous
    orig_data_folder = os.path.join(data_root, mouse, protocol, 'Session Data')

    # better to save directly to scratch and then rsync the saved data over, for now at least
    # foldername_tosave = os.path.join(os.environ['SCRATCH'], 'uchida_lab', 'alowet', 'behavior-plots', mouse, day)
    # foldername_tosave =  os.path.join('/home/adam/Documents/dist-rl/data/behavior-plots/', mouse)
    foldernames_tosave = [os.path.join(x, mouse, day) for x in plot_roots]
    # print(foldername_tosave)

    is_on_cluster = on_cluster()
    if is_on_cluster:
        scratch_data_folder = os.path.join(os.environ['SCRATCH'], 'uchida_lab', 'alowet', 'behavior', mouse, protocol)
        check_dir(scratch_data_folder)
        # scratch_sessions = glob.glob(os.path.join(scratch_data_folder, 'Session Data','*' + day + '*.mat'))
        # store_sessions = glob.glob(os.path.join(orig_data_folder, 'Session Data','*' + day + '*.mat'))
        # if len(scratch_sessions) != len(store_sessions):
        subprocess.call(['rsync', '-avx', '--progress', orig_data_folder, scratch_data_folder])
        beh_data_folder = os.path.join(scratch_data_folder, 'Session Data')
    else:
        beh_data_folder = orig_data_folder

    all_sessions = get_beh_session(beh_data_folder, day)

    for sess in all_sessions:
        infile_name = sess[0]
        session_data = sess[1]
        sess_time = sess[2]

        # kludge for old naming scheme
        if 'imaging' in session_data:
            session_data['has_imaging'] = session_data['imaging']
        # kludge for now. revisit if I ever have multiple repos
        session_data['repo'] = 'Adam'

        infile_path = os.path.join(orig_data_folder, infile_name)

        if 'nTrials' not in session_data or session_data['nTrials'] <= 1:
            print('Too few trials in ' + infile_path + '. You should delete this session from the file system.')
            continue  # don't try to run any analysis here b/c it will break

        print('Plotting session: ' + infile_path)

        # # don't analyze sessions that are of quality 1
        # if 'Quality' in session_data and session_data['Quality'] == 1:
        #     continue
        print(mouse, protocol, day, sess_time)
        filename_tosave = mouse + '_' + protocol + '_' + day + '_' + sess_time

        # make each panel approximately equally sized across figures
        fig_single_session_1 = plt.figure(figsize=(3 * n_trial_types, 2 * 5))
        grid = plt.GridSpec(5, n_trial_types)  # , width_ratios=[.1]+[1]*n_trial_types)
        fig_single_session_2, axs_single_session_2 = plt.subplots(3, 3, figsize=(13, 10))

        ### Plot general parameters of the behavioral session in 2 figures ###

        foreperiod_duration = np.full(session_data['nTrials'], np.nan)
        trace_duration = np.full(session_data['nTrials'], np.nan)
        iti_duration = np.full(session_data['nTrials'], np.nan)
        for i in range(session_data['nTrials']):
            foreperiod_duration[i] = session_data['RawEvents']['Trial'][i].States.Foreperiod[-1]
            trace_duration[i] = session_data['RawEvents']['Trial'][i].States.Trace[-1] - \
                                session_data['RawEvents']['Trial'][i].States.Trace[0]
            iti_duration[i] = session_data['RawEvents']['Trial'][i].States.ITI[-1] - \
                              session_data['RawEvents']['Trial'][i].States.ITI[0]

        plt.figure(fig_single_session_1.number)

        # First row of figure is session info
        plt.subplot(5, n_trial_types, 1)
        plt.hist(session_data['TrialTypes'], bins=np.linspace(0.5, n_trial_types + .5, n_trial_types + 1))
        plt.title('Trial Types', weight='bold')
        plt.ylabel('Ntrials: ' + str(session_data['nTrials']))
        plt.xticks(range(1, n_trial_types + 1), trial_type_names, fontsize=7)
        plt.xlim((0.5, 6.5))

        plt.subplot(5, n_trial_types, 2)
        plt.plot(session_data['TrialTypes'], session_data['RewardDelivered'][:session_data['nTrials']], marker='o',
                 ms=2., ls='')
        plt.xticks(range(1, n_trial_types + 1), trial_type_names, fontsize=7)
        plt.xlim((0.5, 6.5))
        plt.title(r'Reward size ($\mu$L)', weight='bold')

        plt.subplot(5, n_trial_types, 3)
        plt.hist(foreperiod_duration, 5)
        plt.title('Foreperiod dur (s)', weight='bold')

        plt.subplot(5, n_trial_types, 4)
        plt.hist(trace_duration, bins=5, range=(0.5, 5.5))
        plt.title('Trace dur (s)', weight='bold')

        plt.subplot(5, n_trial_types, 5)
        plt.hist(iti_duration, 5)
        plt.title('ITI dur (s)', weight='bold')

        plt.subplot(5, n_trial_types, 6)
        plt.plot(range(session_data['nTrials']), session_data['TrialTypes'], marker='.', ms=2., ls='')
        plt.title('Presented Trials', weight='bold')
        plt.yticks(range(1, n_trial_types + 1))

        ### Following panels: trial-type-specific info and behaviour ###
        time = np.arange(-6., 30., 1. / sr)
        pcolor_time_toplot = np.arange(-1.8, 8., 1. / sr)
        time_toplot = (pcolor_time_toplot[1:] + pcolor_time_toplot[:-1]) / 2  # midpoints
        pos1_time = np.argmin(np.abs(time - time_toplot[0]))
        pos2_time = np.argmin(np.abs(time - time_toplot[-1]))
        # TODO: don't hardcode the 1 (= stim_dur + odor_dur - min_foreperiod_dur = 1 + 2 - 2)
        unexpected_start_time = np.argmin(np.abs(time - 1))

        # get indices for different trial types
        n_trials = session_data['nTrials']
        trial_types = session_data['TrialTypes']
        trial_inds_all_types = [np.flatnonzero(trial_types == i + 1) for i in
                                range(n_trial_types)]  # add 1 because Matlab is 1-indexed
        active_types = [i for i in range(len(trial_inds_all_types)) if len(trial_inds_all_types[i]) > 0]
        n_active_types = len(active_types)

        trace_inds_all_types = [np.flatnonzero(trial_types == i + 1) for i in
                                range(n_trace_types)]  # add 1 because Matlab is 1-indexed
        active_trace_types = [i for i in range(len(trace_inds_all_types)) if len(trace_inds_all_types[i]) > 0]
        n_active_trace_types = len(active_trace_types)

        # structures for storing data. This should work assuming that trace types always come before other types
        mean_licking_pattern = np.full((n_trial_types, len(time)), np.nan)
        sem_licking_pattern = np.full((n_trial_types, len(time)), np.nan)
        mean_licking_last_half_sec = np.full(n_trace_types, np.nan)
        sem_licking_last_half_sec = np.full(n_trace_types, np.nan)
        mean_total_licking_traceperiod = np.full(n_trace_types, np.nan)
        sem_total_licking_traceperiod = np.full(n_trace_types, np.nan)
        aux = 0  # all_auxs[0]  # This will be used for plotting in Fig2, and gets overwritten on each iteration, so reset it here

        half_sec_trial_means = [[]] * n_trial_types
        licks_all_trace_means = [[]] * n_active_trace_types
        licks_groups = [[]] * n_active_trace_types
        TT_all_reward_sizes = [[]] * n_active_trace_types

        mean_licking_pattern_by_size = [[]] * n_active_trace_types
        sem_licking_pattern_by_size = [[]] * n_active_trace_types

        mean_licking_pattern_by_stim = [[]] * n_active_trace_types
        sem_licking_pattern_by_stim = [[]] * n_active_trace_types
        stim_colors = ['k', '#00FFFF']
        n_stims = len(stim_colors)

        mean_licking_pattern_by_stim_loc = [[]] * n_active_trace_types
        sem_licking_pattern_by_stim_loc = [[]] * n_active_trace_types
        stim_loc_colors = ['k', '#7ce8ff', '#55D0FF', '#0080BF']
        n_stim_locs = len(stim_loc_colors)

        licks_all, licks_raw, cs_in_all, cs_out_all, trace_start_all, trace_end_all = \
            getEventTraces(protocol, sr, session_data, range(n_trials), trial_types, time, 'Lick', plot_auxfigs)
        # NaN out licks before 2 seconds before reward in unexpected reward trials
        for exclude_idx in protocol_info['exclude_tt']:
            if exclude_idx < n_trial_types:
                licks_all[trial_inds_all_types[exclude_idx], :unexpected_start_time] = np.nan
                licks_raw[trial_inds_all_types[exclude_idx], :unexpected_start_time] = np.nan

        prc_range = np.nanpercentile(licks_all, [2.5, 97.5])  # great example of why it's best to analyze all trials together in one go!

        tracestart_samples = int(np.flatnonzero(np.round(time, 4) == np.round(np.amax(trace_start_all), 4)))
        traceend_samples = int(np.flatnonzero(np.round(time, 4) == np.round(np.amax(trace_end_all), 4)))

        var_rew_cnt = 0
        for i, this_type in enumerate(active_types):

            trial_inds_this_type = trial_inds_all_types[this_type]
            n_trials_this_type = len(trial_inds_this_type)

            # Second row of figure is the distribution of rewards in the non 100%
            # type of trials (2-4) and the mean lick rate during the trace period
            plt.figure(fig_single_session_1.number)
            # plt.subplot(5, n_trial_types, this_type + n_trial_types + 1)
            # ax = axs_single_session_1.flat[this_type + n_trial_types + 1]
            ax = plt.subplot(grid[1, this_type])
            rew_bins = np.arange(-0.5, np.nanmax(session_data['RewardDelivered'] + 1))
            rew_sizes = np.arange(0, np.nanmax(session_data['RewardDelivered'] + 1))
            heights = np.histogram(session_data['RewardDelivered'][trial_inds_this_type], bins=rew_bins, density=True)
            ax.bar(rew_sizes, heights[0], fc=colors[this_type])
            ax.set_xlim(-1, np.nanmax(session_data['RewardDelivered'] + 1))
            ax.set_ylim(0, 1)
            ax.set_xticks(rew_sizes)
            ax.set_title(trial_type_names[this_type], weight='bold')
            if i == 0: ax.set_ylabel('Probability')
            ax.set_xlabel(r'Reward dist. ($\mu$L)')

            # Third row licking for all trials in the session divided by trial type
            # and fourth row the mean trajectory of the licking for each trial type
            licks_this_type = licks_all[trial_inds_this_type, :]
            mean_licking_pattern[i, :] = np.mean(licks_this_type, 0)
            sem_licking_pattern[i, :] = np.std(licks_this_type, 0) / np.sqrt(n_trials_this_type)

            # plotting
            plt.figure(fig_single_session_1.number)
            ax = plt.subplot(grid[2:4, this_type])
            im = plt.pcolormesh(pcolor_time_toplot, np.arange(n_trials_this_type + 1), licks_this_type[:,
                                                                                       pos1_time:pos2_time + 1],
                                vmin=prc_range[0], vmax=prc_range[1], cmap='magma')
            if i == 0:
                plt.ylabel('Trial')
            if trial_type_names[i] == 'Unexpected':
                plt.axvline(x=0, color=vline_color)
            else:
                for j, trial_ind in enumerate(
                        trial_inds_this_type):  # allows variable CS and trace durations in principle
                    plt.axvline(x=cs_in_all[trial_ind], ymin=j, ymax=j + 1, color=vline_color)
                    plt.axvline(x=cs_out_all[trial_ind], ymin=j, ymax=j + 1, color=vline_color)
                    plt.axvline(x=trace_end_all[trial_ind], ymin=j, ymax=j + 1, color=vline_color)
            plt.title(trial_type_names[this_type], weight='bold')
            plt.ylim(n_trials_this_type, 0)

            avg_ax = fig_single_session_1.add_subplot(4, n_trial_types, 3 * n_trial_types + this_type + 1, sharex=ax)
            trace_dict = {'cs_in': cs_in_all,
                          'cs_out': cs_out_all,
                          'trace_end': trace_end_all,
                          'time_toplot': time_toplot,
                          'pos1_time': pos1_time,
                          'pos2_time': pos2_time,
                          'colors': colors[this_type],
                          'xlim': (-2, 8),
                          'ylabel': 'Lick rate (Hz)',
                          'xlabel': 'Time from CS (s)'
                          }
            if i > 0:
                trace_dict['ylabel'] = ''
            if trial_type_names[i] == 'Unexpected':
                # trace_dict['xlabel'] = 'Time from US'
                # trace_dict['cs_in'] = 0
                trace_dict['cs_out'] = 0
                # trace_dict['trace_end'] = cs_in_all

            plotLickingTrace(trace_dict, mean_licking_pattern[i, :], sem_licking_pattern[i, :], avg_ax)
            setUpLickingTrace(trace_dict, avg_ax)

            # In Figure 2, put the lick traces in different trials types all together for comparison
            plt.figure(fig_single_session_2.number)

            if trial_type_names[this_type] != 'Unexpected':

                # plot avg lick trace for all trial types on one axis
                trace_dict['xlim'] = (-0.5, 8)
                trace_dict['ylabel'] = 'Lick rate (Hz)'
                setUpLickingTrace(trace_dict)
                plotLickingTrace(trace_dict, mean_licking_pattern[i, :], sem_licking_pattern[i, :],
                                 axs_single_session_2[0, 0])
                plt.legend(itemgetter(*active_trace_types)(trial_type_names))
                if i == n_active_trace_types - 1:
                    setUpLickingTrace(trace_dict)

                # For stats after all trials have been run:
                licks_all_trace_means[i] = np.mean(licks_this_type[:, tracestart_samples:traceend_samples + 1], axis=1)
                licks_groups[i] = np.ones((len(licks_all_trace_means[i], ))) * this_type

                # grand mean and sem of last half second of trace period
                half_sec_trial_means[this_type] = np.mean(
                    licks_this_type[:, int(traceend_samples - 0.5 * sr):traceend_samples + 1], axis=1)
                mean_licking_last_half_sec[this_type] = np.mean(half_sec_trial_means[this_type], 0)
                sem_licking_last_half_sec[this_type] = np.std(half_sec_trial_means[this_type], 0) / np.sqrt(
                    n_trials_this_type)

                # Sum of the lick vector during the entire trace period
                total_licking_traceperiod = np.sum(
                    licks_raw[trial_inds_this_type, tracestart_samples:traceend_samples + 1], axis=1)
                mean_total_licking_traceperiod[this_type] = np.mean(total_licking_traceperiod)
                sem_total_licking_traceperiod[this_type] = np.std(total_licking_traceperiod) / np.sqrt(
                    n_trials_this_type)

                # Divide the trials that deliver multiple reward sizes
                TT_all_reward_sizes[i] = np.unique(session_data['RewardDelivered'][trial_inds_this_type])
                n_rew_sizes = len(TT_all_reward_sizes[i])
                colors_sub = np.tile(np.arange(0, 0.8, 1. / (len(TT_all_reward_sizes[i]) + 1)), (3, 1)).T

                mean_licking_pattern_by_size[i] = np.full((n_rew_sizes, np.size(mean_licking_pattern, 1)), np.nan)
                sem_licking_pattern_by_size[i] = np.full((n_rew_sizes, np.size(mean_licking_pattern, 1)), np.nan)

                if has_opto:  # plot stim and nonstim for each trial type
                    mean_licking_pattern_by_stim[i] = np.full((n_stims, np.size(mean_licking_pattern, 1)), np.nan)
                    sem_licking_pattern_by_stim[i] = np.full((n_stims, np.size(mean_licking_pattern, 1)), np.nan)
                    for i_stim in range(n_stims):
                        this_stim_ind = np.logical_and(trial_types == i + 1, session_data['StimTrials'] == i_stim)
                        mean_licking_pattern_by_stim[i][i_stim] = np.mean(licks_all[this_stim_ind, :], axis=0)
                        sem_licking_pattern_by_stim[i][i_stim] = stats.sem(licks_all[this_stim_ind, :], axis=0)
                        trace_dict['colors'] = stim_colors[i_stim]
                        plotLickingTrace(trace_dict, mean_licking_pattern_by_stim[i][i_stim, :],
                                         sem_licking_pattern_by_stim[i][i_stim, :], axs_single_session_2[1, i])
                    if i == 0: plt.legend(['No Stim', 'Stim (All Locs)'])
                    plt.title(trial_type_names[i], weight='bold')
                    setUpLickingTrace(trace_dict)

                    mean_licking_pattern_by_stim_loc[i] = np.full((n_stim_locs, np.size(mean_licking_pattern, 1)), np.nan)
                    sem_licking_pattern_by_stim_loc[i] = np.full((n_stim_locs, np.size(mean_licking_pattern, 1)), np.nan)
                    for i_stim in range(n_stim_locs):
                        this_stim_loc_ind = np.logical_and(trial_types == i + 1, session_data['StimLocs'] == i_stim)
                        mean_licking_pattern_by_stim_loc[i][i_stim] = np.mean(licks_all[this_stim_loc_ind, :], axis=0)
                        sem_licking_pattern_by_stim_loc[i][i_stim] = stats.sem(licks_all[this_stim_loc_ind, :], axis=0)
                        trace_dict['colors'] = stim_loc_colors[i_stim]
                        plotLickingTrace(trace_dict, mean_licking_pattern_by_stim_loc[i][i_stim, :],
                                         sem_licking_pattern_by_stim_loc[i][i_stim, :], axs_single_session_2[2, i])
                    if i == 0: plt.legend(['No Stim', 'Ventral', 'Intermediate', 'Dorsal'])
                    plt.title(trial_type_names[i], weight='bold')
                    setUpLickingTrace(trace_dict)

                elif var_rew_cnt < 3:  # this used to just be else, and is kind of kludgy
                    if n_rew_sizes > 1:
                        for j in range(n_rew_sizes):
                            this_rew_size_ind = np.flatnonzero(
                                session_data['RewardDelivered'][trial_inds_this_type] == TT_all_reward_sizes[i][j])
                            mean_licking_pattern_by_size[i][j, :] = np.mean(licks_this_type[this_rew_size_ind, :], 0)
                            sem_licking_pattern_by_size[i][j, :] = np.std(licks_this_type[this_rew_size_ind, :],
                                                                          0) / np.sqrt(len(this_rew_size_ind))
                            trace_dict['colors'] = colors_sub[j, :]
                            plotLickingTrace(trace_dict, mean_licking_pattern_by_size[i][j, :],
                                             sem_licking_pattern_by_size[i][j, :], axs_single_session_2[1, var_rew_cnt])
                        plt.legend(["{}{}".format(int(k), 'uL') for k in TT_all_reward_sizes[i]])
                        plt.title(trial_type_names[i], weight='bold')
                        setUpLickingTrace(trace_dict)
                        var_rew_cnt += 1

                    else:
                        mean_licking_pattern_by_size[i][0, :] = mean_licking_pattern[i, :]
                        sem_licking_pattern_by_size[i][0, :] = sem_licking_pattern[i, :]
                        if this_type in variable_rew_css:
                            fig_single_session_2.delaxes(axs_single_session_2[1, var_rew_cnt])
                            var_rew_cnt += 1

                    for j in range(len(TT_all_reward_sizes[i])):
                        for amt in rew_code.keys():
                            rew_ind = np.flatnonzero(TT_all_reward_sizes[i][j] == amt)
                            if len(rew_ind) > 0:
                                flag = 1
                                rew_size = amt
                                aux = rew_code[amt]
                                break
                            else:
                                flag = 0
                        if flag:
                            trace_dict['colors'] = colors[this_type]
                            setUpLickingTrace(trace_dict, axs_single_session_2[2, aux])
                            plotLickingTrace(trace_dict, mean_licking_pattern_by_size[i][rew_ind, :].flatten(),
                                             sem_licking_pattern_by_size[i][rew_ind, :].flatten(),
                                             axs_single_session_2[2, aux])
                            plt.title(str(rew_size) + r' $\mu$L', weight='bold')

        flat_licks_all_trace_means = np.concatenate(licks_all_trace_means).ravel()
        flat_licks_half_sec_means = np.concatenate(half_sec_trial_means).ravel()
        flat_licks_groups = np.concatenate(licks_groups).ravel()

        # plot colorbar for pcolor plots
        add_cbar(fig_single_session_1, im, 'Lick rate (Hz)')

        # Mean anticipatory licking 0.5s before US
        plt.subplot(3, 3, 2)
        plt.bar(range(n_trace_types), mean_licking_last_half_sec, yerr=sem_licking_last_half_sec, color=colors,
                align='center', ecolor='black')
        plt.ylim(0, 10)
        plt.xlim(-0.5, n_trace_types - .5)
        plt.xticks(np.arange(n_trace_types), trace_type_names)
        plt.ylabel('Avg. lick rate during last 0.5s (Hz)')
        if protocol == 'DistributionalRL_6Odours':
            plt.text(0 - 0.2, 9, r'$\mu$=0', fontsize='small');
            plt.text(0 - 0.1, 8.2, r'$\sigma^2$=0', fontsize='small')
            plt.text(1 - 0.2, 9, r'$\mu$=4', fontsize='small');
            plt.text(1 - 0.05, 8.2, r'$\sigma^2$=0.8', fontsize='small')
            plt.text(2 - 0.2, 9, r'$\mu$=4', fontsize='small');
            plt.text(2 - 0.1, 8.2, r'$\sigma^2$=4', fontsize='small')
            plt.text(3 - 0.2, 9, r'$\mu$=4', fontsize='small');
            plt.text(3 - 0.1, 8.2, r'$\sigma^2$=4', fontsize='small')
            plt.text(4 - 0.2, 9, r'$\mu$=2', fontsize='small');
            plt.text(4 - 0.1, 8.2, r'$\sigma^2$=0', fontsize='small')
            plt.text(5 - 0.2, 9, r'$\mu$=6', fontsize='small');
            plt.text(5 - 0.1, 8.2, r'$\sigma^2$=0', fontsize='small')

        # Sum of the lick vector during the entire trace period
        plt.subplot(3, 3, 3)
        plt.bar(range(n_trace_types), mean_total_licking_traceperiod, yerr=sem_total_licking_traceperiod, color=colors,
                align='center', ecolor='black')
        plt.xlim(-0.5, n_trace_types - .5)
        plt.xticks(np.arange(n_trace_types), trace_type_names)
        plt.ylabel('Avg. total licking during trace')

        # Compare same reward amounts in different CS conditions
        # for i, this_type in enumerate(active_trace_types):
            # for j in range(len(TT_all_reward_sizes[i])):
            #     for amt in rew_code.keys():
            #         rew_ind = np.flatnonzero(TT_all_reward_sizes[i][j] == amt)
            #         if len(rew_ind) > 0:
            #             flag = 1
            #             rew_size = amt
            #             aux = rew_code[amt]
            #             break
            #         else:
            #             flag = 0
            #     if flag:
            #         trace_dict['colors'] = colors[this_type]
            #         setUpLickingTrace(trace_dict, axs_single_session_2[2, aux])
            #         plotLickingTrace(trace_dict, mean_licking_pattern_by_size[i][rew_ind, :].flatten(),
            #                          sem_licking_pattern_by_size[i][rew_ind, :].flatten(), axs_single_session_2[2, aux])
            #         plt.title(str(rew_size) + r' $\mu$L', weight='bold')

        fig_single_session_1.subplots_adjust(wspace=0.4, hspace=1)
        fig_single_session_2.subplots_adjust(wspace=0.3, hspace=0.5)
        for fig in [fig_single_session_1, fig_single_session_2]:
            for ax in fig.axes:
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')

        # Stats on mean lick rate during the trace period (anticipatory licking)
        # handle case where there is only one trial type, in which case it makes no sense to do stats
        if len(active_trace_types) == 1:
            F, p_anova = np.nan, np.nan
            H, p_kruskal = np.nan, np.nan
            F_half_sec, p_anova_half_sec = np.nan, np.nan
            F_full_trace, p_anova_full_trace = np.nan, np.nan
            H_half_sec, p_kruskal_half_sec = np.nan, np.nan
            H_full_trace, p_kruskal_full_trace = np.nan, np.nan
            tukeyHSD_result, tukeyHSD_result_half_sec, tukeyHSD_result_full_trace = None, None, None
            stat_man_half_sec, p_man_half_sec = np.nan, np.nan
            mean_diff_half_sec = np.nan
        else:
            # gets rid of empty lists without raising deprecation warning
            trim_half_sec_means = [x for x in half_sec_trial_means if type(x) == np.ndarray]
            F_half_sec, p_anova_half_sec = stats.f_oneway(*trim_half_sec_means)
            F_full_trace, p_anova_full_trace = stats.f_oneway(*licks_all_trace_means)
            # mean_diff_half_sec = np.mean(half_sec_trial_means[protocol_info['high_tt']]) - np.mean(
                # half_sec_trial_means[protocol_info['null_tt']])
            half_sec_trial_means = np.array(half_sec_trial_means)
            mean_diff_half_sec = np.mean(np.hstack(half_sec_trial_means[protocol_info['high_tt']])) - \
                                 np.mean(np.hstack(half_sec_trial_means[protocol_info['null_tt']]))
            # to avoid ValueError('All numbers are identical in kruskal')
            # ignore phase 3 of Bernoulli, where there's no 100% CS, because it fails if there is no licking to 0% CS
            if not all([np.all(trial_type_array == trial_type_array[0]) for trial_type_array in licks_all_trace_means]) \
                    and len(active_trace_types) > 2 and len(np.hstack(half_sec_trial_means[protocol_info['high_tt']])) > 0 and \
                    not np.all(np.hstack(half_sec_trial_means[protocol_info['high_tt']]) == 0):

                H_half_sec, p_kruskal_half_sec = stats.kruskal(*trim_half_sec_means, nan_policy='raise')
                H_full_trace, p_kruskal_full_trace = stats.kruskal(*licks_all_trace_means, nan_policy='raise')
                tukeyHSD_half_sec = pairwise_tukeyhsd(flat_licks_half_sec_means, flat_licks_groups)
                tukeyHSD_full_trace = pairwise_tukeyhsd(flat_licks_all_trace_means, flat_licks_groups)

                # JSON can't serialize numpy boolean objects, so we'll need this for later
                tukeyHSD_result_half_sec = [[bool(x) if type(x) == np.bool_ else x for x in sublst] for sublst in
                                            tukeyHSD_half_sec._results_table.data]
                tukeyHSD_result_full_trace = [[bool(x) if type(x) == np.bool_ else x for x in sublst] for sublst in
                                              tukeyHSD_full_trace._results_table.data]

                # print(len(half_sec_trial_means), half_sec_trial_means)
                # print(half_sec_trial_means[protocol_info['null_tt']])
                # print(half_sec_trial_means[protocol_info['high_tt']])
                # [print(np.sum(np.isnan(x))) for x in half_sec_trial_means]
                stat_man_half_sec, p_man_half_sec = stats.mannwhitneyu(np.hstack(half_sec_trial_means[protocol_info['null_tt']]),
                                                                       np.hstack(half_sec_trial_means[protocol_info['high_tt']]))
            else:
                H_half_sec, p_kruskal_half_sec = np.nan, np.nan
                H_full_trace, p_kruskal_full_trace = np.nan, np.nan
                tukeyHSD_result_half_sec, tukeyHSD_result_full_trace = None, None
                stat_man_half_sec, p_man_half_sec = np.nan, np.nan

        # save statistics of lick RATE (not total licks) for both the last half second and the full trace period (2 sec)
        summ_stats = {'F_half_sec': F_half_sec,
                      'F_full_trace': F_full_trace,
                      'p_anova_half_sec': p_anova_half_sec,
                      'p_anova_full_trace': p_anova_full_trace,
                      'H_half_sec': H_half_sec,
                      'H_full_trace': H_full_trace,
                      'p_kruskal_half_sec': p_kruskal_half_sec,
                      'p_kruskal_full_trace': p_kruskal_full_trace,
                      'tukeyHSD_result_half_sec': tukeyHSD_result_half_sec,
                      'tukeyHSD_result_full_trace': tukeyHSD_result_full_trace,
                      'stat_man_half_sec': stat_man_half_sec,
                      'p_man_half_sec': p_man_half_sec,
                      'mean_diff': mean_diff_half_sec
                      }

        if savefiles:
            # save figures as PNG and vector graphic
            for foldername_tosave in foldernames_tosave:
                check_dir(foldername_tosave)
                save_path = os.path.join(foldername_tosave, filename_tosave)
                fig_single_session_1.savefig(save_path + '_1.png', format='png', bbox_inches='tight', dpi=300)
                fig_single_session_2.savefig(save_path + '_2.png', format='png', bbox_inches='tight', dpi=300)

            # save important vars
            pickle_save_path = os.path.join(foldernames_tosave[0], filename_tosave)
            save_dict = {'stats': summ_stats, 'licks_smoothed': licks_all, 'licks_raw': licks_raw,
                         'active_types': active_types, 'trial_types': trial_types, 'time': time, 'sr': sr}
            with open(os.path.join(pickle_save_path + '.p'), 'wb') as f:
                pickle.dump(save_dict, f)

            try:
                db_data = select_db(db['db'], 'session', '*', 'name = ? AND protocol = ? AND exp_date = ? AND exp_time = ?',
                                    (mouse, protocol, day, sess_time))
                db_dict = {k: db_data[k] for k in db_data.keys()}
            except:  # when I couldn't access db frome x
                with open(db['config'], 'rb') as f:
                    config = json.load(f)
                db_keys = [x[0] for x in config['session_fields']]
                db_dict = {k: session_data[k] for k in db_keys if k in session_data}

                db_data = execute_sql(f'SELECT mid, sid, rid, exp_date FROM session WHERE name="{mouse}" AND protocol="{protocol}" ORDER BY sid DESC, rid DESC', db['db'])
                for entry in db_data:
                    mid = entry[0]
                    exp = entry[3]
                    if exp == int(session_data['exp_date']):
                        sid = entry[1]
                        rid = entry[2] + 1
                        break
                    elif exp < int(session_data['exp_date']):
                        sid = entry[1] + 1
                        rid = 1
                        break
                    else:
                        sid = 1
                        rid = 1
                db_dict['mid'] = mid; db_dict['sid'] = sid; db_dict['rid'] = rid


            if int(day) <= 20201118:
                # old naming convention before database unification
                old_keys = ['image', 'n_trials', 'notes', 'file_date', 'file_time']
                new_keys = ['has_imaging', 'n_trial', 'session_cmt', 'exp_date', 'exp_time']
                for old_key, new_key in zip(old_keys, new_keys):
                    session_data[new_key] = session_data[old_key]
                # add required keys mid, sid, and rid
            for k in db_dict.keys():
                # don't overwrite session comment from database, and don't use raw_data_path from file
                if k in session_data and k not in ['session_cmt', 'raw_data_path']:
                    db_dict[k] = session_data[k]

            db_dict['stats'] = json.dumps(summ_stats)
            db_dict['significance'] = int(p_man_half_sec < alpha and mean_diff_half_sec > 0.75)

            if db_dict['session_cmt'] is not None and 'WARNING' in db_dict['session_cmt']:  # if frames dropped
                db_dict['exclude'] = 1

            if is_on_cluster:
                db_dict['figure_path'] = foldernames_tosave[0]
                db_dict['date_analyzed'] = datetime.today().strftime('%Y%m%d')
                db_dict['raw_data_path'] = infile_path
                insert_into_db(db['db'], 'session', tuple(db_dict.keys()), tuple(db_dict.values()))

            else:
                # don't update paths from local computer. Don't update date_analyze
                db_dict['git_hash'] = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
                insert_into_db(db['db'], 'session', tuple(db_dict.keys()), tuple(db_dict.values()))

        else:
            plt.show()

        print('Finished plotting session: ' + infile_path + '. Saved to ' + save_path)


if __name__ == '__main__':
    import argparse

    # defined command line options
    # this also generates --help and error handling
    CLI = argparse.ArgumentParser()

    # CLI.add_argument('data_root')
    CLI.add_argument('mouse')
    CLI.add_argument('protocol')
    CLI.add_argument('day')
    # CLI.add_argument(
    #     "--plot_roots",  # name on the CLI - drop the `--` for positional/required parameters
    #     nargs=2,  # 0 or more values expected => creates a list
    #     type=str
    # )
    CLI.add_argument(
        "--savefiles", '-s',
        type=int,
        default=1
    )
    CLI.add_argument("--has_opto", '-o', type=int, default=0)

    # parse the command line
    args = CLI.parse_args()
    plotSingleSession(args.mouse, args.protocol, args.day, args.savefiles, args.has_opto)
    # plotSingleSession(args.data_root, args.mouse, args.protocol, args.day, args.plot_roots, args.savefiles)
