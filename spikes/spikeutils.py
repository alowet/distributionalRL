import glob
import os
import json
import numpy as np
from scipy.interpolate import interp1d
import sys

sys.path.append('../utils')
from matio import loadmat
from paths import raise_print
from plotting import validate_timestamps
from db import select_db


def get_ephys_session(beh_data_folder, day, time=None):
    session = glob.glob(beh_data_folder + '/*' + day + '*.mat')
    datafile_names = []
    ephys_sessions = []
    file_times = []
    if len(session) == 0:
        raise_print('Could not find mat files in ' + beh_data_folder)
    for df in session:
        converted_data = loadmat(os.path.join(beh_data_folder, df))
        session_data = converted_data['SessionData']
        if session_data['has_ephys'] == 1:
            # # if Bpod gets funky, sometimes this is returned as an empty list instead of an int. If it freezes, the field doesn't exist at all
            # if 'quality' in session_data and isinstance(session_data['quality'], int):
            datafile_names.append(df)
            ephys_sessions.append(session_data)
            file_times.append(session_data['exp_time'])
    if len(datafile_names) == 1:
        return datafile_names[0], ephys_sessions[0]
    elif len(datafile_names) > 1:
        # figure out which (behavior) file_time is closest to (ephys) meta_time, and use that behavior file
        which_session = np.argmin(np.abs([time - int(file_time) for file_time in file_times]))
        return datafile_names[which_session], ephys_sessions[which_session]
    else:
        raise_print("Couldn't find ephys session on this day.")


def get_timestamps(session_data, n_trace_types, n_trials):

    timestamps = {'foreperiod': 1,  # seconds before stimulus start to plot
                  'iti': 2,
                  # seconds after reward to plot. In later versions of the protocol, this is called PostRew, not ITI
                  'align': np.full(n_trials, np.nan),
                  'trial_start': np.full(n_trials, np.nan),
                  'ttl_high': np.full(n_trials, np.nan),  # TTL goes HIGH at start of foreperiod
                  'ttl_low': np.full(n_trials, np.nan),  # TTL gos LOW at start of ITI
                  'trace': np.full(n_trials, np.nan),
                  'stim': np.full(n_trials, np.nan)}

    for i in range(n_trials):
        timestamps['align'][i] = session_data['RawEvents']['Trial'][i].States.Foreperiod[-1]
        timestamps['trial_start'][i] = session_data['TrialStartTimestamp'][i]
        timestamps['ttl_high'][i] = session_data['TrialStartTimestamp'][i] + \
                                    session_data['RawEvents']['Trial'][i].States.Foreperiod[0]
        timestamps['ttl_low'][i] = session_data['TrialStartTimestamp'][i] + \
                                   session_data['RawEvents']['Trial'][i].States.ITI[0]
        timestamps['trace'][i] = session_data['RawEvents']['Trial'][i].States.Trace[-1] - \
                                 session_data['RawEvents']['Trial'][i].States.Trace[0]
        if session_data['TrialTypes'][i] < n_trace_types + 1:
            stimulus_field = getattr(session_data['RawEvents']['Trial'][i].States,
                                     'Stimulus' + str(session_data['TrialTypes'][i]) + 'Delivery')
            timestamps['stim'][i] = stimulus_field[-1] - stimulus_field[0]
        elif session_data['TrialTypes'][i] == n_trace_types + 1:
            timestamps['stim'][i] = np.nan
        else:
            raise_print('Trial type not recognized.')
    timestamps['ttls'] = np.sort(np.concatenate((timestamps['ttl_high'], timestamps['ttl_low'])))
    timestamps['bin'] = 0.001  # 1 ms

    timestamps = validate_timestamps(timestamps)
    return timestamps


def convert_spike_times(spike_samples, timestamps, db_dict):
    """
	:param spike_times: vector of all spike times in the SpikeGLX timebase
	:param timestamps: output of get_timestamps
	:param db_dict: dictionary of this entry in the ephys db
	:return: vector of all spike times in the Bpod timebase
	"""
    n_trials = len(timestamps['ttl_high'])

    ttl_events = json.loads(db_dict['ttl_events']) if not db_dict['cut_short'] else json.loads(db_dict['ttl_events_used'])
    ttl_times = np.array(ttl_events[0]) / db_dict['samp_rate']
    spike_times_glx = spike_samples / db_dict['samp_rate']

    # Bpod won't save the last trial if it is interrupted by a stop signal, so there's a good chance SpikeGLX registers
    # an extra pair of TTLs at the end, which we will disregard. In the event that the session is cut_short (e.g. due
    # to electrical disconnection), then there may be an odd number of ttls registered during the good period, with
    # length n_trials * 2 + 1
    assert n_trials * 2 <= len(ttl_times) <= (n_trials + 1) * 2, \
        "Incorrect number of TTL events in SpikeGLX file. Are you sure you started Bpod after neural data, and ended " \
        "Bpod before neural data?"
    # more error checking
    assert np.all(np.array(ttl_events[1][::2]) == 1), "Every other TTL must be positive"
    assert np.all(np.array(ttl_events[1][1::2]) == -1), "Every other TLL must be negative"

    # trim to correct length
    ttl_times = ttl_times[:n_trials * 2]
    f = interp1d(ttl_times, timestamps['ttls'], kind='linear', fill_value='extrapolate', assume_sorted=True)
    spike_times = f(spike_times_glx)

    return spike_times


def chunk_trials(cell_spikes, timestamps, exclude_trials):
    n_trials = len(timestamps['ttl_high'])
    # will contain one entry for each spike, aligned within the trial
    spikes_by_trial = np.empty(n_trials, dtype=object)

    # for the full matrix, used for smoothing and average psth
    time_per_trial = timestamps['foreperiod'] + np.amax(timestamps['stim']) + np.amax(timestamps['trace']) + \
                     timestamps['iti']
    trial_timebase = np.arange(-timestamps['foreperiod'], time_per_trial - timestamps['foreperiod'], timestamps['bin'])
    sample_timebase = np.arange(0, (time_per_trial / timestamps['bin']))

    # interpolate times into samples, now aligned per trial
    f = interp1d(trial_timebase, sample_timebase, kind='linear', assume_sorted=True)

    # preallocate for speed
    spike_mat = np.zeros((n_trials, int(time_per_trial / timestamps['bin'])), dtype=bool)

    for i in range(n_trials):

        start_aligned = timestamps['trial_start'][i] + timestamps['align'][i] - timestamps['foreperiod']
        end_aligned = timestamps['trial_start'][i] + timestamps['align'][i] + timestamps['stim_trial'][i] + \
                      timestamps['trace_trial'][i] + timestamps['iti']
        if exclude_trials[i]:  # unexpected reward, so need to shift it to trial end instead of trial start
            # start_aligned = timestamps['trial_start'][i] + timestamps['align'][i] - timestamps['foreperiod'] - \
            #                 timestamps['stim'] - timestamps['trace']
            # end_aligned = timestamps['trial_start'][i] + timestamps['align'][i] + timestamps['iti']
            # So that when I smooth, it doesn't plummet to zero
            # end_aligned = timestamps['trial_start'][i] + timestamps['align'][i] + timestamps['stim_trial'][i] + \
            #               timestamps['trace_trial'][i] + timestamps['iti'] + 1
            # offset = timestamps['stim'] + timestamps['trace']
            start_aligned -= (timestamps['stim'] + timestamps['trace'])
        # else:
            # end_aligned = timestamps['trial_start'][i] + timestamps['align'][i] + timestamps['stim_trial'][i] + \
            #               timestamps['trace_trial'][i] + timestamps['iti']
            # offset = 0
        spikes_by_trial[i] = cell_spikes[np.logical_and(cell_spikes > start_aligned,
                                                        cell_spikes < end_aligned - timestamps['bin'])] - \
                             start_aligned - timestamps['foreperiod']
        spike_inds = np.around(f(spikes_by_trial[i])).astype(np.int16)
        spike_mat[i, spike_inds] = 1

    return spikes_by_trial, spike_mat


def bin_session(cell_spikes, db_entry, timestamps):
    """
	When calculating z-scored firing rate, we want to take the entire contiguous session, not broken up by trial
	:param cell_spikes:
	:return:
	"""
    spike_bins = np.around(cell_spikes / db_entry['samp_rate'] / timestamps['bin']).astype(np.int64)
    n_bins = int(np.around(db_entry['recording_dur'] / timestamps['bin'])) + 1
    recording_mat = np.zeros(n_bins)
    recording_mat[spike_bins] = 1
    return recording_mat


def get_putative_coords(dbdict, paths):
    ret = select_db(paths['db'], 'session', '*', 'raw_data_path=?', (dbdict['behavior_path'],))
    coords = ['probe1_AP', 'probe1_ML', 'probe1_DV', 'probe1_angle', 'probe1_dye']
    for coord in coords:
        dbdict[coord] = ret[coord]
    # TODO: in future, define a precise mapping between subregions and channels based on the allen CCF
    if 0.4 <= ret['probe1_AP'] <= 1.7 and 0.5 <= abs(ret['probe1_ML']) <= 2.1:
        dbdict['probe1_region'] = 'striatum'
    elif 2.5 < ret['probe1_AP'] < 3. and 0. < abs(ret['probe1_ML']) < 1.5:
        dbdict['probe1_region'] = 'PFC'
    else:
        raise Exception('Probe region not recognized at AP {} and abs ML {}'.format(ret['probe1_AP'], ret['probe1_ML']))
    return dbdict
