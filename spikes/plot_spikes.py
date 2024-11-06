import numpy as np
from scipy.stats import sem, variation, kurtosis
from scipy.ndimage import gaussian_filter1d
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import warnings

import sys
import os

from spikeutils import *
from classify_units import classify_units

sys.path.append('../utils')
from plotting import *
from paths import *
from db import *
from matio import *
from protocols import load_params

sys.path.append('../behavior_analysis')
from traceUtils import setUpLickingTrace


def plot_spikes(data_path, plot=1, rerun=0, plot_unexpected=1, highlights=[]):
    """
    :param data_path: Path to spike_times.npy, spike_clusters.npy, and cluster_group.tsv, formatted as
    /path/to/data/mouse_name/file_date_id/ks_matlab
    :return:
    """
    plt.style.use('paper_export')
    # print(plot)
    paths, db_dict, session_data, protocol, names_tosave, pupil, behavior = analyze_neural(data_path, 'ephys')
    # print(db_dict)

    # protocol-specific info
    # colors, _, trial_type_names, trace_type_names, n_trial_types, n_trace_types, _ = get_cs_info(protocol)
    # exclude_tt = trial_type_names.index('Unexpected')
    colors, protocol_info, periods, kwargs = load_params(protocol)
    colors = colors['colors']
    trial_type_names = protocol_info['trial_type_names']
    trace_type_names = protocol_info['trace_type_names']
    n_trial_types = protocol_info['n_trial_types']
    n_trace_types = protocol_info['n_trace_types']
    exclude_tt = protocol_info['exclude_tt']

    # load processed neural data from Kilosort/Phy
    neuron_path = os.path.join(data_path, 'ks_matlab')
    spike_samples = np.load(os.path.join(neuron_path, 'spike_times.npy'))
    spike_clusters = np.load(os.path.join(neuron_path, 'spike_clusters.npy'))

    if db_dict['cut_short']:
        # spike_duration = (spike_samples[-1] - spike_samples[0]) / db_dict['samp_rate']
        # behavior_duration = session_data['TrialEndTimestamp'][-1] - session_data['TrialStartTimestamp'][0]
        # ttl_times = np.array(json.loads(db_dict['ttl_events'])[0]) / db_dict['samp_rate']
        print('This session was cut short')
        n_ttls = np.sum(spike_samples[-1] > np.array(json.loads(db_dict['ttl_events'])[0]))
        n_trials = int(n_ttls // 2)
        db_dict['ttl_events_used'] = json.dumps([x[:n_ttls] for x in json.loads(db_dict['ttl_events'])])

        samps_to_use = (spike_samples < db_dict['recording_dur'] * db_dict['samp_rate']).flatten()
        spike_samples = spike_samples[samps_to_use]
        spike_clusters = spike_clusters[samps_to_use]

    else:
        n_trials = len(session_data['RawEvents']['Trial'])
        # db_dict['ttl_events_used'] = db_dict['ttl_events']
    db_dict['n_trials_used'] = n_trials

    # load and parse the tsv to get the indices of good clusters
    # tsv_file = open(os.path.join(neuron_path, 'cluster_group.tsv'))
    # cluster_group = np.loadtxt(tsv_file, dtype=str, delimiter='\t', skiprows=1)
    # good_cells = cluster_group[cluster_group[:, 1] == 'good', 0].astype(np.int32)

    # classify good units into cell types, and return good units cluster_ids
    good_cells = classify_units(db_dict['name'], db_dict['file_date_id'], db_dict['ncells'], rerun=rerun)
    # print(good_cells)
    print(len(good_cells))

    timestamps = get_timestamps(session_data, n_trace_types, n_trials)
    spike_times_bpod = convert_spike_times(spike_samples, timestamps, db_dict)
    trial_types = session_data['TrialTypes'][:n_trials] - 1
    # might have to change later if stim/trace durations change
    max_x = timestamps['stim'] + timestamps['trace'] + timestamps['iti']
    box_ys = np.insert(np.cumsum([np.sum(trial_types == i) for i in range(n_trial_types)]), 0, 0)
    time = np.arange(-timestamps['foreperiod'], max_x, timestamps['bin'])
    timestamps['time'] = time
    # adjust timebase to account for pcolor behavior
    pcolor_time = np.arange(-timestamps['foreperiod'] - (timestamps['bin'] / 2), max_x + (timestamps['bin'] / 2),
                            timestamps['bin'])

    # get behavior times to plot based on neural times to plot
    behavior['time'], behavior['start'], behavior['end'] = get_timebase(time, behavior['dat']['time'],
                                                                        1. / behavior['dat']['sr'], 3)

    if pupil:
        pupil['time'], pupil['start'], pupil['end'] = get_timebase(time, pupil['dat']['timebase'],
                                                                   pupil['dat']['timebase'][1] -
                                                                   pupil['dat']['timebase'][0], 2)

    # Gaussian kernel for smoothing PSTH
    sigma_s = 0.100  # 100 ms
    sigma = int(sigma_s / timestamps['bin'])
    rect_width = .1

    # for saving means
    n_cells = len(good_cells)
    trial_type_means = np.full((n_cells, n_trial_types, len(time)), np.nan)
    session_means = np.zeros(n_cells)
    session_stds = np.zeros(n_cells)
    session_stability = np.zeros(n_cells)
    session_kurtosis = np.zeros(n_cells)

    if plot:
        pdfname_tosave = names_tosave['filenames'][0] + '_spikes.pdf'
        pdf = PdfPages(pdfname_tosave)

    time_per_trial = timestamps['foreperiod'] + np.amax(timestamps['stim']) + np.amax(timestamps['trace']) + \
                     timestamps['iti']
    all_spikemats = np.zeros((n_cells, n_trials, int(time_per_trial / timestamps['bin'])), dtype=bool)
    firing = np.zeros((n_cells, n_trials, int(time_per_trial / timestamps['bin'])))

    trace_dict = {'ylabel': 'Mean firing rate (Hz)', 'xlabel': 'Time from CS (s)', 'xlim': (time[0], time[-1] + rect_width*2),
                  'cs_in': 0, 'cs_out': 1, 'trace_end': 3}

    for i, i_cell in enumerate(good_cells):

        print('Analyzing cell {}'.format(i))

        # get complete firing rate trace for that neuron over the entire recording, for z-scoring
        # do this for each cell individually in case it requires too much memory
        binned = bin_session(spike_samples[spike_clusters == i_cell], db_dict, timestamps)
        smoothed = gaussian_filter1d(binned.astype(np.float64), sigma) / timestamps['bin']
        session_means[i] = np.mean(smoothed)  # also equal to len(cell_spikes) / db_dict['recording_dur']
        session_stds[i] = np.std(smoothed)

        # find coefficient of variation across 10 equally-spaced chunks to test stability
        bins = np.linspace(spike_samples[0, 0], spike_samples[-1, 0], 11)
        hist, _ = np.histogram(spike_samples[spike_clusters == i_cell], bins)
        session_stability[i] = variation(hist)
        session_kurtosis[i] = kurtosis(hist)

        # get spikes aligned to trial events and sort by trial type
        cell_spikes = spike_times_bpod[spike_clusters == i_cell]
        spikes_by_trial, spike_mat = chunk_trials(cell_spikes, timestamps, trial_types == exclude_tt)
        all_spikemats[i] = spike_mat
        spikes_by_trial_sorted = spikes_by_trial[np.argsort(trial_types, kind='mergesort')]

        spike_mat_smoothed = gaussian_filter1d(spike_mat.astype(np.float64), sigma, axis=1) / timestamps['bin']
        firing[i] = spike_mat_smoothed
        for j in range(n_trial_types):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trial_type_means[i, j] = np.nanmean(spike_mat_smoothed[trial_types == j], axis=0)

        if plot:

            # plot rasters, broken up by trial type
            fig, axs = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
            ax = axs[0]
            setUpLickingTrace(trace_dict, ax, override_ylims=True)
            if plot_unexpected:
                n_plot_types = n_trial_types
                n_plot_trials = n_trials
                plot_type_names = trial_type_names
            else:
                n_plot_types = n_trace_types
                n_plot_trials = np.sum(trial_types != exclude_tt)
                spikes_by_trial_sorted = spikes_by_trial_sorted[:n_plot_trials]
                plot_type_names = trace_type_names

            rects = [Rectangle((max_x + rect_width, box_ys[j]), rect_width, box_ys[j + 1] - box_ys[j]) for j in
                     range(n_plot_types)]
            ax.set_ylim([n_plot_trials, -1])
            ax.eventplot(spikes_by_trial_sorted, color='k', linewidths=1)
            # print(colors)
            pc = PatchCollection(rects, facecolors=colors)
            ax.add_collection(pc)
            ax.set_clip_on(False)
            ax.axis('off')
            # joining the x-axes forces the x labels to be the same, so just turn this all off and label Trials
            # by hand
            # ax.set_ylabel('Trials')
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.spines['bottom'].set_color('none')
            # ax.spines['left'].set_color('none')

            # plot mean/sem during the trial
            ax = axs[1]
            # exclude_tt_len = int((timestamps['foreperiod'] + timestamps['iti']) / timestamps['bin'])
            # spike_mat_smoothed[trial_types == exclude_tt, exclude_tt_len:] = np.nan
            handles = [0] * n_plot_types
            setUpLickingTrace(trace_dict, ax, override_ylims=True)

            for j in range(n_plot_types):
                handles[j] = ax.plot(time, trial_type_means[i, j], c=colors[j], lw=1)
            ax.legend(plot_type_names, loc=(1.04, 0))  # call legend first so that it doesn't try to label shading

            for j in range(n_plot_types):
                spike_sem = sem(spike_mat_smoothed[trial_types == j], axis=0)
                ax.fill_between(time, trial_type_means[i, j] - spike_sem, trial_type_means[i, j] + spike_sem,
                                color=colors[j], edgecolor=None, alpha=0.2)

            ax.get_shared_x_axes().join(ax, axs[0])
            ax.set_xticks([0, 2, 4])
            fig.suptitle('Mouse {m}, session {d},\nNeuron {n}'.format(m=db_dict['name'], d=db_dict['file_date_id'], n=i))
            # ax.set_ylabel('Mean firing rate (Hz)')
            # ax.set_xlabel('Time from CS (s)')
            # ax.set_xticks(np.arange(time[0], time[-1]))
            # add_vlines(fig, timestamps['stim'] + timestamps['trace'])
            # ax.axvspan(xmin=0, xmax=timestamps['stim'], alpha=.5, color=(.8, .8, .8),
            #            label='_nolegend_')
            hide_spines()
            plt.subplots_adjust(wspace=0, hspace=0)
            pdf.savefig(fig, bbox_inches='tight')
            if i in highlights:
                plt.show(block=False)
            plt.close()

    if plot:
        pdf.close()
        subprocess.call(['rsync', '-avx', '--progress', pdfname_tosave, names_tosave['foldernames'][1]])

    # get z-scored firing rate
    zscore_fr = (trial_type_means - session_means[:, np.newaxis, np.newaxis]) / session_stds[:, np.newaxis, np.newaxis]
    plot_all_neurons(zscore_fr, behavior, pupil, trial_types, protocol, timestamps, pcolor_time, db_dict, names_tosave,
                     'FR (std)', 'spikes')

    # save data
    trial_inds_all_types = [np.flatnonzero(trial_types == i) for i in range(n_trial_types)]
    cached_data = {'spks': all_spikemats, 'timestamps': timestamps, 'trial_types': trial_types,
                   'trial_inds_all_types': trial_inds_all_types, 'means': session_means, 'stds': session_stds,
                   'cvs': session_stability, 'kurtosis': session_kurtosis, 'firing': firing}
    save_pickle(names_tosave['filenames'], cached_data, 'spikes')
    np.save(os.path.join(data_path, 'ks_matlab', 'spike_times_bpod.npy'), spike_times_bpod)

    # record in database that session has been processed
    # if on_cluster():
    #     db_dict = get_putative_coords(db_dict, paths)
    #     insert_into_db(paths['db'], 'ephys', tuple(db_dict.keys()), tuple(db_dict.values()))
    #     print('Inserted ephys database information for ' + names_tosave['filenames'][0])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot spikes')
    parser.add_argument('data_path', type=str)
    parser.add_argument('-p', '--plot', type=int, default=1)
    parser.add_argument('-r', '--rerun', type=int, default=0)
    parser.add_argument('-u', '--unexpected', type=int, default=1)
    args = parser.parse_args()
    plot_spikes(args.data_path, args.plot, args.rerun, args.unexpected)

    # if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
    #     plot_spikes(sys.argv[1])
    # else:
    #     raise Exception('Invalid path specified. Usage: python plot_spikes.py data_path')
