import os
import matplotlib
# matplotlib.use('Agg')
import numpy as np
from scipy.stats import zscore, sem, pearsonr
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import cmocean
import pickle
from neuralUtils import *
from suite2p.extraction import dcnv
import time
import warnings
import subprocess
import glob
from PyPDF2 import PdfReader

import sys

sys.path.append('../utils')
from plotting import *
from paths import *
from db import *
from protocols import *

sys.path.append('../behavior_analysis')
from traceUtils import setUpLickingTrace

def check_file(fullfile):
    with open(fullfile, 'rb') as f:
        try:
            pdf = PdfReader(f)
            info = pdf.metadata
            if info:
                return True
            else:
                return False
        except Exception as e:
            return False

def plotTrialAvgsByNeuron(video_dir, plot=1):

    plt.style.use('paper_export')
    # general parameters
    paths, db_dict, session_data, protocol, names_tosave, pupil, behavior = analyze_neural(video_dir, 'imaging')

    # plt.figure()
    # last_high = [session_data['RawEvents']['Trial'][i].Events.BNC1High[-1] for i in range(len(session_data['RawEvents']['Trial']))]
    # iti_end = [session_data['RawEvents']['Trial'][i].States.ITI[-1] for i in range(len(session_data['RawEvents']['Trial']))]
    # plt.scatter(last_high, iti_end)
    # plt.show()

    check_integrity(db_dict)  # make sure session is curated and transferred
    temp_dir = get_temp_dir(db_dict, video_dir)

    # no matter if it's continuous or not, create a scratch directory (if on cluster) in order to make writing pdfs much faster
    pdf_dir = get_pdf_dir(db_dict, video_dir)
    check_dir(os.path.join(pdf_dir, 'tmp'))
    pdf_root = os.path.join(pdf_dir, 'tmp', db_dict['name'] + '_' + db_dict['file_date_id'])

    # protocol-specific info
    colors, _, trial_type_names, _, n_trial_types, n_trace_types, _ = get_cs_info(protocol)
    exclude_tt = trial_type_names.index('Unexpected')

    # load processed neural data from suite2p
    neuron_path = os.path.join(temp_dir, 'suite2p', 'plane0')
    ops = np.load(os.path.join(neuron_path, 'ops.npy'), allow_pickle=True)
    ops = ops.item()

    n_chan = ops['nchannels']
    try:
        chan_names = ops['chan_names']
        if len(chan_names) != n_chan:
            print('Using default config: Channels are [Ca, DA]')
            chan_names = ['Ca', 'DA']  # if nchannels were saved differently than the code, default to this
    except KeyError:  # kludge
        chan_names = ['DA']

    if db_dict['meta_date'] > 20220406 and db_dict['continuous'] == 1:
        fudge = .0153
    else:
        fudge = 0

    chan1 = np.load(os.path.join(neuron_path, 'F.npy'))  # channel 1
    Fneu1 = np.load(os.path.join(neuron_path, 'Fneu.npy'))
    chans = [(chan1, Fneu1)]
    if n_chan == 2:
        chan2 = np.load(os.path.join(neuron_path, 'F_chan2.npy'))  # channel 2
        Fneu2 = np.load(os.path.join(neuron_path, 'Fneu_chan2.npy'))
        chans.append((chan2, Fneu2))
    spks = np.load(os.path.join(neuron_path, 'spks.npy'))
    stat = np.load(os.path.join(neuron_path, 'stat.npy'), allow_pickle=True)

    iscell = np.load(os.path.join(neuron_path, 'iscell.npy'))
    cell_inds = np.flatnonzero(iscell[:, 0])
    n_cells = np.size(cell_inds, 0)

    # show what this FOV looked like
    compute_ROIs_and_footprints(ops, stat, cell_inds, names_tosave['filenames'])

    # get trial info from behavior file
    n_trials_behavior = session_data['nTrials']
    print('Getting timestamps for mouse {}, day {}'.format(db_dict['name'], db_dict['file_date_id']))
    if db_dict['continuous']:
        # confirm that it was in fact continuous acquisition
        # if on_cluster():
        tiff_lens, metadata_fs = count_scanimage_tiffs(temp_dir, i_tiffs=[0])
        if tiff_lens[0] % 1000 != 0:  # with 2 channels it will be 2000 frames per TIFF, not 1000
            raise_print(
                'Length of first TIFF was not equal to 1000. Are you sure this was acquired in continuous mode?')
        timestamps = get_timestamps(session_data, n_trials_behavior, n_trace_types, metadata_fs, fudge=fudge)
        # plt.hist(np.diff(timestamps['tiff_starts']))
        # plt.figure()
        # plt.scatter(session_data['TrialStartTimestamp'], timestamps['tiff_starts'])
        # print(pearsonr(session_data['TrialStartTimestamp'], timestamps['tiff_starts']))
        # plt.figure()
        # plt.scatter(session_data['TrialEndTimestamp'][:-1], timestamps['tiff_starts'][1:])
        # print(pearsonr(session_data['TrialEndTimestamp'][:-1], timestamps['tiff_starts'][1:]))
        # plt.show()
        n_trials = timestamps['last_trial'] - timestamps['first_trial']
        fill_val = 'extrapolate'


    else:
        # load image data
        # print('Loading TIFFs in ' + temp_dir)
        # total_tiffs = count_scanimage_tiffs(temp_dir)
        tiff_lens, metadata_fs = count_scanimage_tiffs(temp_dir, i_tiffs=None)
        # tiff_counts = np.floor_divide(total_tiffs, n_chan)
        # these should be the same, but take the minimum just in case, and print a warning
        n_trials_imaging = len(tiff_lens)
        # if behavior is stopped mid-trial, then microscope will have begun, but trial never completed, so don't count it
        if not (n_trials_imaging == n_trials_behavior or n_trials_imaging == n_trials_behavior + 1):
            warnings.warn('Imaging and Behavior trial numbers do not agree. Using smaller of the two.')
        n_trials = np.amin([n_trials_behavior, n_trials_imaging])
        timestamps = get_timestamps(session_data, n_trials, n_trace_types, meta_fs=metadata_fs, tiff_counts=tiff_lens)
        fill_val = np.nan

    # compute actual frame rate from TTLs
    avg_fs = np.nanmean(timestamps['fs'])
    # compare against the frame rate in the table
    if np.abs(avg_fs - db_dict['fs']) > 1:
        raise Exception(
            'Empirical frame rate is {} but database says {}. Revise database entry manually.'.format(avg_fs, db_dict['fs']))

    n_samps = int((timestamps['foreperiod'] + timestamps['stim'] + timestamps['trace'] + timestamps['iti']) * avg_fs)
    # because pcolormesh puts values between the endpoints
    pcolor_time = np.linspace(-timestamps['foreperiod'], timestamps['stim'] + timestamps['trace'] + timestamps['iti'],
                              num=n_samps + 1)
    time = (pcolor_time[:-1] + pcolor_time[1:]) / 2  # midpoints of pcolor_time, since pcolor labels both endpoints
    align_ind = np.argmin(np.abs(time))  # time is already aligned! Don't subtract off timestamps['foreperiod'] here
    timestamps['time'] = time  # for saving later

    # get behavior times to plot based on neural times to plot
    behavior['time'], behavior['start'], behavior['end'] = get_timebase(time, behavior['dat']['time'],
                                                                        1. / behavior['dat']['sr'], 3)

    if pupil:
        pupil['time'], pupil['start'], pupil['end'] = get_timebase(time, pupil['dat']['timebase'],
                                                                   pupil['dat']['timebase'][1] -
                                                                   pupil['dat']['timebase'][0], 2)

    # get indices corresponding to trials of this type
    trial_types = session_data['TrialTypes'][timestamps['first_trial']:timestamps['last_trial']] - 1
    trial_inds_all_types = [np.flatnonzero(trial_types == i) for i in range(n_trial_types)]
    active_types = [i for i in range(n_trial_types) if len(trial_inds_all_types[i]) > 0]
    # n_active_types = len(active_types)

    rewards = session_data['RewardDelivered'][timestamps['first_trial']:timestamps['last_trial']].astype(np.int16)
    combos = np.array(['_'.join([str(x), str(y)]) for x, y in zip(trial_types, rewards)], dtype='object')
    combo_types = sorted(np.unique(combos))
    combo_tts = [int(x.split('_')[0]) for x in combo_types]
    combo_rews = [y.split('_')[1] for y in combo_types]
    combo_names = [trial_type_names[x] + ' ' + y + '$\mu$L' for x, y in zip(combo_tts, combo_rews)]
    combo_colors = [colors[x] for x in combo_tts]
    reward_cmap = mpl.cm.copper
    reward_norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(rewards))
    reward_colors = reward_cmap([float(x)/np.amax(rewards) for x in combo_rews])

    # Gaussian kernel for smoothing deconv activty
    sigma_s = 0.100  # 100 ms
    sigma = int(np.around(sigma_s * avg_fs))
    print(sigma)

    # for raster plots
    rect_width = .1
    max_x = timestamps['stim'] + timestamps['trace'] + timestamps['iti']
    trace_dict = {'ylabel': '', 'xlabel': 'Time from CS (s)',
                  'xlim': (pcolor_time[0], pcolor_time[-1] + rect_width * 3.5),
                  'cs_in': 0, 'cs_out': 1, 'trace_end': 3}

    deconv_info = {'abbr': ['deconv', 'zF', 'dFF'],
                   'cbar_label': ['Activity', 'Fluorescence (std)', r'$\Delta F/F$']}

    for chan_num, (Ftot, Fneu) in enumerate(chans):

        # subtract neuropil fluorescence
        Fout = Ftot - ops['neucoeff'] * Fneu

        # baseline operation
        # Fc = dcnv.preprocess(Fout, baseline=ops['baseline'], win_baseline=200, sig_baseline=5, fs=ops['fs'], prctile_baseline=ops['prctile_baseline'])
        # Fc = dcnv.preprocess(Fout, baseline='constant_percentile', win_baseline=ops['win_baseline'],
        #                      sig_baseline=ops['sig_baseline'], fs=ops['fs'], prctile_baseline=ops['prctile_baseline'])

        # declare variables now, for saving later
        spks_store = None
        spks_smooth = None
        zF_store = None
        dFF_store = None

        for i_deconv, deconv in enumerate([True, False, False]):

            if deconv:  # equivalent to spks.npy
                # I tested this with the baseline method used here; deconvolution works way better with the maximin
                # method that is suite2p's default, as does zF
                # this_activity = dcnv.oasis(F=Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])
                # if db_dict['continuous']:
                #     this_activity = spks.copy()
                # else:
                #     continue  # don't try deconvolution if acquisition was not continuous
                this_activity = spks.copy()  # do it regardless of continuous acquisition
                this_smooth_activity = gaussian_filter1d(this_activity.astype(np.float64), sigma) * avg_fs
            elif i_deconv == 1:  # z-scored dF/F
                # we should z-score the whole thing before packing into the 3D-arrays
                Fc = dcnv.preprocess(Fout, baseline=ops['baseline'], win_baseline=ops['win_baseline'],
                                     sig_baseline=ops['sig_baseline'], fs=ops['fs'],
                                     prctile_baseline=ops['prctile_baseline'])
                this_activity = zscore(Fc, axis=1)
                this_smooth_activity = this_activity
            else:
                # for dF/F, constant_percentile method is better because it avoids negative values for baseline subtraction
                this_activity = dcnv.preprocess(Fout, baseline='constant_percentile', win_baseline=ops['win_baseline'],
                                                sig_baseline=ops['sig_baseline'], fs=ops['fs'],
                                                prctile_baseline=ops['prctile_baseline'])
                this_smooth_activity = this_activity

            # set up data structures for storage
            activity = np.full((n_cells, n_trials, n_samps), np.nan)  # trailing axis must have len(time) for subtraction
            smooth_activity = np.full((n_cells, n_trials, n_samps), np.nan)
            mean_activity = np.full((n_cells, n_trial_types, n_samps), np.nan)
            combo_activity = np.full((n_cells, len(combo_types), n_samps), np.nan)

            # get the frame closest to the target start time, which is timestamps['foreperiod'] seconds before stimulus onset
            # this is in a timebase within a single trial, starting from 0
            start_times = timestamps['align'].copy()  # zero is odor onset
            start_times[np.isin(trial_types, exclude_tt)[timestamps['first_trial']:timestamps['last_trial']]] -= \
                (timestamps['stim'] + timestamps['trace'])

            # get the frame closest to the target end time, which is timestamps['iti'] seconds after reward onset
            # this is in a timebase within a single trial, starting from 0
            # end_times = timestamps['align'] + timestamps['stim_trial'] + timestamps['trace_trial'] + timestamps['iti']

            for i, i_trial in enumerate(range(timestamps['first_trial'], timestamps['last_trial'])):

                # start_idx = (np.abs(timestamps['frame_midpoints'][i_trial] - start_times[i_trial])).argmin()
                # end_idx = (np.abs(timestamps['frame_midpoints'][i_trial] - end_times[i_trial])).argmin()
                # neural_timebase = timestamps['frame_midpoints'][i_trial][start_idx:end_idx + 1] - timestamps['align'][
                #     i_trial]

                # this should account for Unexpected Reward trials nicely, unless they are the very first trial, in
                # which case I make them NaN
                f = interp1d(timestamps['frame_midpoints'][i_trial] - start_times[i_trial],  # zero is odor onset
                             np.arange(len(timestamps['frame_midpoints'][i_trial])), 'linear', bounds_error=False, fill_value=fill_val)
                tiff_pos = f(time)
                tiff_inds_to_use = np.around(timestamps['tiff_starts'][i_trial] + tiff_pos[~np.isnan(tiff_pos)]).astype(np.int32)
                activity[:, i, ~np.isnan(tiff_pos)] = this_activity[cell_inds][..., tiff_inds_to_use]
                smooth_activity[:, i, ~np.isnan(tiff_pos)] = this_smooth_activity[cell_inds][..., tiff_inds_to_use]
                if len(tiff_inds_to_use) == n_samps:
                    activity[:, i, tiff_inds_to_use < 0] = np.nan
                    smooth_activity[:, i, tiff_inds_to_use < 0] = np.nan
                # cut off more of the trial so that deconv artifacts don't screw things up
                if deconv and trial_types[i_trial] == exclude_tt and not db_dict['continuous']:
                    activity[:, i, time < timestamps['stim'] + sigma_s*2] = np.nan
                    smooth_activity[:, i, time < timestamps['stim'] + sigma_s*2] = np.nan

                # use 'nearest' interpolation to align these samples to the standard timebase
                # F_all_cells_neural_timebase = this_activity[cell_inds, int(np.round(timestamps['tiff_starts'][i_trial] + start_inc)):int(np.round(timestamps['tiff_starts'][i_trial] + end_inc)) + 1]

                # try:
                #     # f = interp1d(neural_timebase, F_all_cells_neural_timebase, 'nearest', axis=1,
                #     #              bounds_error=False)  # , fill_value='extrapolate'
                #     activity[:, i, :] = f(time)
                # except ValueError:  # the number of tiffs doesn't match up with the number recorded by bpod, e.g. b/c the PMTs shut off and I deleted the blanks
                #     activity[:, i, :] = np.nan

            if deconv:
                spks_store = activity.copy()
                spks_smooth = smooth_activity.copy()
            elif i_deconv == 1:
                zF_store = activity.copy()
            elif i_deconv == 2:
                # consider baseline F as the foreperiod duration median. Make it nanmedian because for some old looped
                # acquisitions, there is less than one second of baseline
                baselines = np.nanmean(activity[:, :, :align_ind], axis=2, keepdims=True)
                # baselines = np.nanpercentile(activity[:, :, :align_ind], 10., axis=2, keepdims=True)
                dFF_store = (activity - baselines) / baselines
                activity = dFF_store.copy()

            if plot:

                pdfname_tosave = pdf_root + '_chan_' + chan_names[chan_num] + '_' + deconv_info['abbr'][
                    i_deconv] + '_neurons'
                suffixes = ['', '_reward', '_raster', '_reward_raster']

                if not deconv:  # replot deconv using shading
                    if np.all([os.path.isfile(pdfname_tosave + x + '.pdf') for x in suffixes]):
                        # and np.all([time.strptime(time.ctime(os.path.getmtime(pdfname_tosave + x + '.pdf'))) >= time.strptime('20231006', '%Y%m%d') for x in suffixes]):
                        print('File exists and is recent')
                        if np.all([check_file(pdfname_tosave + x + '.pdf') for x in suffixes]):
                            print('File is valid. Skipping {} for channel {}, mouse {}, day {}'.format(
                                deconv_info['abbr'][i_deconv], chan_names[chan_num], db_dict['name'],
                                db_dict['file_date_id']))
                            continue
                            # print('Not skipping')
                        else:
                            print('File is corrupt/incomplete')

                [pdf, pdfrew, raster, rasterrew] = [PdfPages(pdfname_tosave + x + '.pdf') for x in suffixes]

                print('Plotting {} for channel {}, mouse {}, day {}'.format(
                    deconv_info['abbr'][i_deconv], chan_names[chan_num], db_dict['name'], db_dict['file_date_id']))

                for i_cell in range(n_cells):

                    print(db_dict['name'], db_dict['file_date_id'], i_cell, '/', n_cells)

                    # prc_range = np.percentile(zdF[i_cell, :, :], [2.5, 97.5], axis=None)
                    prc_range = np.nanpercentile(activity[i_cell, :, :], [2.5, 97.5], axis=None)

                    handles = [0] * n_trial_types
                    for ref, iter, iter_mean, iter_names, iter_colors, iter_pdfs in zip(
                            [trial_types, combos], [range(n_trial_types), combo_types], [mean_activity, combo_activity],
                            [trial_type_names, combo_names], [colors, combo_colors], [(pdf, raster), (pdfrew, rasterrew)]):

                        # set up axes
                        n_rows = 2
                        n_plot_types = len(iter)
                        single_trial_fig, single_trial_axes = plt.subplots(n_rows, n_plot_types, figsize=(n_plot_types*2.5, 5),
                                                                           gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
                        set_share_axes(single_trial_axes[1, :], sharey=True)
                        # del_unused_axes(single_trial_fig, single_trial_axes, active_types)

                        # set up raster
                        raster_fig, raster_axes = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
                        box_ys = np.insert(np.cumsum([np.sum(this_type == ref) for this_type in iter]), 0, 0)

                        ax = raster_axes[0]
                        ax.set_ylim([n_trials, -1])
                        activity_by_trial_sorted = activity[i_cell, np.argsort(ref, kind='mergesort'), :]
                        this_cmap = plt.cm.gray_r if deconv else cmocean.tools.crop(cmocean.cm.balance, prc_range[0], prc_range[1], pivot=0)
                        masked = np.ma.masked_invalid(activity_by_trial_sorted)
                        im = ax.pcolormesh(pcolor_time, range(n_trials + 1), masked, vmin=prc_range[0], vmax=prc_range[1], cmap=this_cmap)

                        cbar_ax = raster_fig.add_axes([0.98, 0.65, .02, .2])
                        cbar = raster_fig.colorbar(im, cax=cbar_ax)
                        cbar.set_label(deconv_info['cbar_label'][i_deconv], rotation=270, labelpad=15)

                        rects = [Rectangle((max_x + rect_width, box_ys[j]), rect_width, box_ys[j + 1] - box_ys[j]) for j in
                                 range(n_plot_types)]
                        pc = PatchCollection(rects, facecolors=iter_colors)
                        ax.add_collection(pc)
                        if ref is combos:
                            rewrects = [Rectangle((max_x + rect_width*2.5, box_ys[j]), rect_width, box_ys[j + 1] - box_ys[j]) for j in range(n_plot_types)]
                            pc = PatchCollection(rewrects, facecolors=reward_colors)
                            ax.add_collection(pc)
                            cax = raster_fig.add_axes([1.25, 0.65, .02, .2])
                            cbar = raster_fig.colorbar(mpl.cm.ScalarMappable(norm=reward_norm, cmap=reward_cmap), cax=cax, ticks=[0, 2, 4, 6])
                            cbar.set_label('Reward size ($\mu$L)', rotation=270, labelpad=15)

                        ax.set_clip_on(False)
                        ax.axis('off')
                        # joining the x-axes forces the x labels to be the same, so just turn this all off and label Trials
                        # by hand
                        # ax.set_ylabel('Trials')
                        # ax.set_yticks([])
                        # ax.set_xticks([])
                        # ax.spines['bottom'].set_color('none')
                        # ax.spines['left'].set_color('none')
                        if deconv:
                            ax.axvspan(0, 1,  alpha=.5, facecolor=(.8, .8, .8), label='_nolegend_')
                        else:
                            ax.axvline(0, c='k', lw=1)
                            ax.axvline(1, c='k', lw=1)
                        ax.axvline(3, c='k', lw=1)

                        # for i, trial_type in enumerate(active_types):
                        for i_type, this_type in enumerate(iter):
                            # trial_type_inds = trial_inds_all_types[i_type]
                            trial_type_inds = np.flatnonzero(this_type == ref)
                            n_trials_this_type = len(trial_type_inds)
                            # trial_type_inds_beh = trial_type_inds + timestamps['first_trial']
                            trial_type_inds_beh = trial_type_inds + timestamps['first_trial']

                            # take mean across trials of this type for this cell
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', category=RuntimeWarning)
                                # mean_zdF[i_cell, i, :] = np.nanmean(zdF[i_cell, trial_type_inds, :], axis=0)
                                # if there are any nan timepoints, exclude from the average so it doesn't get distorted
                                iter_mean[i_cell, i_type, :] = np.mean(smooth_activity[i_cell, trial_type_inds, :], axis=0)

                            # plot activity with all trial types on the same color axis
                            ax = single_trial_axes[0, i_type]
                            this_cmap = plt.cm.gray_r if deconv else cmocean.tools.crop(cmocean.cm.balance, prc_range[0], prc_range[1], pivot=0)

                            masked = np.ma.masked_invalid(activity[i_cell, trial_type_inds, :])
                            im = ax.pcolormesh(pcolor_time, range(n_trials_this_type + 1), masked,
                                               vmin=prc_range[0], vmax=prc_range[1], cmap=this_cmap)
                            if i_type == 0:
                                ax.set_ylabel('Trial')
                            ax.set_title(iter_names[i_type])
                            ax.set_yticks(range(0, n_trials_this_type, max(n_trials_this_type // 2, 1)))
                            ax.set_ylim(n_trials_this_type, 0)

                            # # compute licks, first converting NaNs to zeros
                            # lick_raster = behavior['dat']['licks_raw'][trial_type_inds_beh, behavior['start']:behavior['end'] + 1]
                            # lick_raster[np.isnan(lick_raster)] = 0
                            # licks_raw_inds_toplot_this_type = np.nonzero(lick_raster)
                            #
                            # # overlay lick raster
                            # if not deconv:
                            #     ax.scatter(x=behavior['time'][licks_raw_inds_toplot_this_type[1]],
                            #                y=licks_raw_inds_toplot_this_type[0] + 0.5, s=3, c='k', alpha=.4)  # , marker='|')

                            ax = single_trial_axes[1, i_type]
                            ax.plot(time, iter_mean[i_cell, i_type, :], c=iter_colors[i_type])

                            # compute and plot sem
                            sem_activity = sem(smooth_activity[i_cell, trial_type_inds, :], axis=0, nan_policy='omit')
                            ax.fill_between(time, iter_mean[i_cell, i_type, :] + sem_activity,
                                            iter_mean[i_cell, i_type, :] - sem_activity,
                                            color=iter_colors[i_type], edgecolor=None, alpha=0.2)
                            # if i_type == exclude_tt:
                            #     ax.set_xlabel('Time from US (s)')
                            # else:
                            ax.set_xlabel('Time from CS (s)')
                            if i_type == 0:
                                ax.set_ylabel('Mean ' + deconv_info['cbar_label'][i_deconv])

                        # plot mean for this cell below raster. Only plot for trial types, not combos, lest it get too confusing
                        for i_type in range(n_trial_types):
                            # plot mean/sem during the trial
                            ax = raster_axes[1]
                            handles[i_type] = ax.plot(time, mean_activity[i_cell, i_type, :], c=colors[i_type], lw=1)

                        ax.legend(trial_type_names, loc=(1.04, 0))  # call legend first so that it doesn't try to label shading
                        setUpLickingTrace(trace_dict, ax, override_ylims=True)  # only do this for PSTH b/c won't show up on raster

                        for i_type in range(n_trial_types):
                            sem_activity = sem(smooth_activity[i_cell, i_type == trial_types, :], axis=0, nan_policy='omit')
                            ax.fill_between(time, mean_activity[i_cell, i_type, :] - sem_activity,
                                            mean_activity[i_cell, i_type, :] + sem_activity,
                                            color=colors[i_type], edgecolor=None, alpha=0.2)

                        ax.get_shared_x_axes().join(ax, raster_axes[0])
                        ax.set_xticks([0, 2, 4])
                        ax.set_ylabel('Mean ' + deconv_info['cbar_label'][i_deconv])
                        raster_fig.suptitle(
                            'Mouse {m}, session {d}, Neuron {n}, {c}'.format(m=db_dict['name'], d=db_dict['file_date_id'],
                                                                             n=i_cell, c=deconv_info['abbr'][i_deconv]))
                        hide_spines()
                        raster_fig.subplots_adjust(wspace=0, hspace=0)
                        iter_pdfs[1].savefig(raster_fig, bbox_inches='tight')
                        plt.close(raster_fig)

                        # adjust plot layout
                        single_trial_fig.suptitle(
                            'Mouse {m}, session {d}, Neuron {n}, {c}'.format(m=db_dict['name'], d=db_dict['file_date_id'],
                                                                             n=i_cell, c=deconv_info['abbr'][i_deconv]))
                        add_cbar_and_vlines(single_trial_fig, im, deconv_info['cbar_label'][i_deconv],
                                            timestamps['stim'] + timestamps['trace'])
                        # exclude_axs=single_trial_axes[:, exclude_tt])
                        iter_pdfs[0].savefig(single_trial_fig)
                        plt.close(single_trial_fig)

                # [x.close() for x in pdf, raster, pdfrew, rasterrew]
                pdf.close()
                raster.close()
                pdfrew.close()
                rasterrew.close()

                [subprocess.call(['rsync', '-avx', '--progress', pdfname_tosave + x + '.pdf', names_tosave['foldernames'][0]]) for x in suffixes]
                # [subprocess.call(['rsync', '-avx', '--progress', pdfname_tosave + x + '.pdf', names_tosave['foldernames'][1]]) for x in suffixes]

                this_cmap = plt.cm.gray_r if deconv else None
                plot_all_neurons(mean_activity, behavior, pupil, trial_types, protocol, timestamps, pcolor_time, db_dict,
                                 names_tosave, deconv_info['abbr'][i_deconv], chan_names[chan_num], cmap=this_cmap,
                                 first_trial=timestamps['first_trial'])

        cached_data = {'F': zF_store, 'spks': spks_store, 'spks_smooth': spks_smooth, 'dFF': dFF_store,
                       'trial_inds_all_types': trial_inds_all_types, 'active_types': active_types,
                       'timestamps': timestamps}
        save_pickle(names_tosave['filenames'], cached_data, chan_names[chan_num])

    # record in database that session has been processed
    if on_cluster():
        insert_into_db(paths['db'], 'imaging', tuple(db_dict.keys()), tuple(db_dict.values()))
        print('Inserted imaging database information for ' + names_tosave['filenames'][0])

    # print(spks_store.shape)
    # plt.show(block=False)
    print('plotTrialAvgsByNeuron completed successfully.')
    return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot imaging')
    parser.add_argument('video_dir', type=str)
    parser.add_argument('-p', '--plot', type=int, default=1)
    args = parser.parse_args()
    plotTrialAvgsByNeuron(args.video_dir, args.plot)
