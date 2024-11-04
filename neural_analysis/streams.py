import numpy as np
import pickle
from scipy.interpolate import interp1d
import os
import sys
sys.path.append('../utils')
from db import select_db, get_db_info
from matio import loadmat


class DataStream():
    """
    This is a class to extract common features from behavior (i.e. licking), neural (i.e. imaging or ephys) and
    facemap data into a common format. In addition to regress_session, it is also used in facemap_decoding.ipynb,
    so change it only with caution.
    """

    def __init__(self, pfile, source, start_time=-1., end_time=5.):
        self.p = pfile
        self.source = source
        self.start_time = start_time
        self.end_time = end_time
        if pfile is None:  # case where there is no facemap data
            self.time = np.arange(-1, 5, 1/30)
            self.end_ind = np.argmin(np.abs(self.time - end_time))
            self.dat = {'dat': {'mot_svd': np.empty(shape=(50, len(self.time), 0))}}
        else:
            self.dat = pickle.load(open(self.p, 'rb'))
            self.mat = None

            if source == 'behavior':
                self.per = 1. / self.dat['sr']
                self.time = self.dat['time']
                self.end_ind = np.argmin(np.abs(self.time - end_time))
            elif source == 'neural':
                self.time = self.dat['timestamps']['time']
                try:  # for spike data
                    self.per = self.dat['timestamps']['bin']
                    # NOTE! It's not ideal that dt is buried right here. This is where the GLM bin width gets set
                    self.dt = 0.020  # for resampling/binning spikes
                    # self.dt = 0.050
                except:  # for 2p data
                    self.per = 1. / np.mean(self.dat['timestamps']['fs'])
                    self.dt = self.per
                self.end_ind = len(self.time)  # because it only goes until 4.999 as of now
                self.rew_time = self.dat['timestamps']['stim'] + self.dat['timestamps']['trace']
                self.rew_ind = np.argmin(np.abs(self.time - self.rew_time))
            elif source == 'facemap':
                self.per = np.mean(np.diff(self.dat['dat']['timebase']))
                self.time = self.dat['dat']['timebase']
                self.end_ind = np.argmin(np.abs(self.time - end_time))
        self.start_ind = np.argmin(np.abs(self.time - start_time))
        self.align_ind = np.argmin(np.abs(self.time))
        self.nsamp = self.end_ind - self.start_ind


def extract_data_streams(name, file_date, file_date_id, table, meta_time=None):

    print(name, str(file_date), file_date_id)

    paths = get_db_info()
    if table == 'ephys':
        ret = select_db(paths['db'], 'session', '*', 'name=? AND exp_date=? AND has_ephys=1', (name, file_date))
    elif table == 'imaging':
        print(paths['db'], name, file_date)
        rets = select_db(paths['db'], 'session', '*', 'name=? AND exp_date=? AND has_imaging=1', (name, file_date), unique=False)
        i_ret = np.argmin([np.abs(ret['exp_time'] - meta_time*100) for ret in rets])
        print(i_ret, rets[i_ret]['exp_time'])
        ret = rets[i_ret]
    behavior_path = os.path.join(paths['behavior_fig_roots'][0], name, str(file_date))
    behavior_p = os.path.join(behavior_path,
                              '_'.join([name, ret['protocol'], str(file_date), str(ret['exp_time']).zfill(6) + '.p']))

    if ret['has_ephys']:
        suffix = 'spikes.p'
    elif ret['has_imaging']:
        suffix = 'Ca.p'

    neural_path = os.path.join(paths['neural_fig_roots'][0], name, file_date_id)
    neural_p = os.path.join(neural_path, '_'.join([name, str(file_date), suffix]))

    if ret['has_facemap'] == 1:
        facemap_path = os.path.join(paths['facemap_root'], name, file_date_id)
        facemap_p = os.path.join(facemap_path, '_'.join([name, str(file_date), 'facemap.p']))
    else:
        facemap_p = None

    raw_behavior_path = ret['raw_data_path'].replace(paths['remote_behavior_root'], paths['behavior_root'])
    raw = loadmat(raw_behavior_path)
    behavior = DataStream(behavior_p, 'behavior')
    neural = DataStream(neural_p, 'neural')
    facemap = DataStream(facemap_p, 'facemap')

    # extract into useful format

    # trim everything down to the same timebase: -1 to 5 s (at least for now; consider extending this in the future)
    # for unexpected reward trials, this should be -1 to 2 s
    behavior.mat = behavior.mat
    behavior.mat_raw = behavior.dat['licks_raw'][:, behavior.start_ind:behavior.end_ind]
    behavior.mat_raw[np.isnan(behavior.mat_raw)] = 0.  # turn NaNs into zeros, for convolution (they'll be trimmed out later)
    neural.mat_raw = neural.dat['spks'][:, :, neural.start_ind:neural.end_ind]

    # downsample both licks and spikes by summing within bins
    for stream in [behavior, neural]:
        stream.down = neural.dt / stream.per
        stream.mat = np.stack(np.array([np.sum(stream.mat_raw[..., int(round(st)):int(round(en))],
                                               axis=-1) for st, en in zip(
            np.arange(0, stream.nsamp - stream.down + 1, stream.down),
            np.arange(stream.down, stream.nsamp + 1, stream.down))]), axis=-1)
    t_fun = interp1d(np.arange(neural.nsamp), neural.time[neural.start_ind:neural.end_ind])
    time = t_fun(np.arange(0, neural.nsamp, neural.down))

    # important for when cut_short = 1
    n_trials = neural.mat.shape[1]
    behavior.mat = behavior.mat[:n_trials]
    behavior.mat_raw = behavior.mat_raw[:n_trials]

    try:
        facemap.mat_raw = np.stack(
            (*[facemap.dat['dat'][m][:, facemap.start_ind:facemap.end_ind] for m in
               ['whisking', 'running', 'pupil'] if m in facemap.dat['dat'].keys()],
             *np.transpose(facemap.dat['dat']['mot_svd'][:, facemap.start_ind:facemap.end_ind, :],
                           (2, 0, 1))
             ), axis=0)
    except ValueError:  # neither whisking, running, nor pupil exists
        facemap.mat_raw = np.transpose(facemap.dat['dat']['mot_svd'][:, facemap.start_ind:facemap.end_ind, :],
                                       (2, 0, 1))
    except KeyError:  # mot_svd doesn't exist
        facemap.mat_raw = np.stack([facemap.dat['dat'][m][:, facemap.start_ind:facemap.end_ind] for m in
                                    ['whisking', 'running', 'pupil'] if m in facemap.dat['dat'].keys()])

    # n_trials important when cut_short = 1
    facemap.mat_raw = facemap.mat_raw[:, :n_trials, :]

    # resample camera metrics to match neural sr
    my_resample = interp1d(facemap.time[facemap.start_ind:facemap.end_ind], facemap.mat_raw,
                           kind='linear', axis=2, fill_value='extrapolate')
    facemap.mat = my_resample(time)

    facemap.labels = ['pupil'] if 'pupil' in facemap.dat['dat'].keys() else []
    if 'mot_svd' in facemap.dat['dat']:
        facemap.labels += ['mot_svd' + str(i) for i in range(facemap.dat['dat']['mot_svd'].shape[-1])]

    # separate out whisking and running for convolution
    n_face_conv = len([x for x in ['whisking', 'running'] if x in facemap.dat['dat'].keys()])
    facemap.conv = facemap.mat[:n_face_conv, ...]
    facemap.nonconv = facemap.mat[n_face_conv:, ...]

    return time, neural, behavior, facemap, raw, ret

