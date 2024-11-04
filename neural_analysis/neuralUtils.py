import numpy as np
import matplotlib.pyplot as plt
import os
from copy import copy
import ScanImageTiffReader
from skimage.color import rgb2hsv, hsv2rgb
import subprocess
from scipy import io
import warnings
import sys

sys.path.append('../utils')
from matio import loadmat
from paths import raise_print, check_dir
from db import get_db_info, on_cluster
from plotting import validate_timestamps

def compute_ROIs_and_footprints(ops, stat, cell_inds, fnames, alpha=0.7, i_cell=None):
    #  adapted from https://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik/9204506

    plt.figure()
    # im = ops['meanImg']/255

    # use max projection, if it exists
    if 'max_proj' in ops:
        mproj = ops['max_proj']
        mimg1 = np.percentile(mproj, 1)
        mimg99 = np.percentile(mproj, 99)
        mproj = (mproj - mimg1) / (mimg99 - mimg1)
        maxproj = np.zeros((ops['Ly'], ops['Lx']), np.float32)
        try:
            maxproj[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]] = mproj
        except:
            print('maxproj not in combined view')
        maxproj = np.maximum(0, np.minimum(1, maxproj))

    meanimg = ops['meanImg']
    mimg1 = np.percentile(meanimg, 1)
    mimg99 = np.percentile(meanimg, 99)
    meanimg = (meanimg - mimg1) / (mimg99 - mimg1)
    meanimg = np.maximum(0, np.minimum(1, meanimg))

    rows, cols = meanimg.shape

    # from CellReg documentation:
    # The matrix of the spatial footprints is of size NxMxK, where N is the number of neurons, M is the number of
    # pixels in the y axis and K is the number of pixels in the x axis.
    footprint = np.zeros((len(cell_inds), rows, cols))
    color_mask = np.zeros((rows, cols, 3))
    color = [255, 178, 0]  # bright orange. Arbitrary choice

    if i_cell is None:
        for i, n in enumerate(cell_inds):
            # color = list(np.random.choice(range(256), size=3))
            ypix = stat[n]['ypix'][~stat[n]['overlap']]
            xpix = stat[n]['xpix'][~stat[n]['overlap']]
            footprint[i, ypix, xpix] = stat[n]['lam'][~stat[n]['overlap']] / np.sum(stat[n]['lam'][~stat[n]['overlap']])
            color_mask[ypix, xpix] = color

        # save footprint
        for fname in fnames:
            io.savemat(fname + '_footprint.mat', {'footprint': footprint})

    else:
        n = cell_inds[i_cell]
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        color_mask[ypix, xpix] = color

    # Construct RGB version of grey-level image
    for mimg, img_name in zip([maxproj, meanimg], ['_max_proj', '_mean']):
        im_color = np.dstack((mimg, mimg, mimg))

        # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            im_hsv = rgb2hsv(im_color)
            color_mask_hsv = rgb2hsv(color_mask)

        # Replace the hue and saturation of the original image with that of the color mask
        im_hsv[..., 0] = color_mask_hsv[..., 0]
        im_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        im_masked = hsv2rgb(im_hsv)

        for plot_mask in ['_mask', '_raw']:

            if plot_mask == '_mask':
                plt.imshow(im_masked)
            else:
                plt.imshow(im_color)

            for fname in fnames:
                # to remove large padding
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                # save the figure
                plt.savefig(fname + img_name + plot_mask + '.png', format='png', bbox_inches='tight', pad_inches=0)

    plt.close()


def check_integrity(db_dict):
    paths = get_db_info()
    if db_dict['curated'] != 1:
        sys.exit('Curated status {} for mouse {}, file date id {}. Skipping'.format(db_dict['curated'], db_dict['name'],
                                                                                    db_dict['file_date_id']))

    # make sure iscell.npy has been transferred already. If it hasn't, then transfer it (or raise an Exception)
    if db_dict['transferred'] != 1:
        iscell_file = os.path.join(paths['home_root'], db_dict['name'], db_dict['file_date_id'], 'suite2p', 'plane0',
                                   'iscell.npy')
        old_file = os.path.join(paths['imaging_root'], db_dict['name'], db_dict['file_date_id'], 'suite2p', 'plane0',
                                'iscell.npy')
        # if file in home directory is newer than the one in 2P-microscope directory
        if on_cluster() and os.path.getmtime(iscell_file) > os.path.getmtime(old_file):
            subprocess.call(['rsync', '-avx', '--progress', iscell_file, os.path.dirname(old_file)])
            db_dict['transferred'] = 1
        else:
            raise_print('iscell.npy has not yet been updated in the imaging root. '
                        'rsync iscell.npy from home directory at /n/home06/alowet/dist-rl/data/mouse_name/file_date_id/plane0')

    if on_cluster():
        # remove files from home directory
        this_path = os.path.join(paths['home_root'], db_dict['name'], db_dict['file_date_id'])
        if os.path.exists(this_path):
            subprocess.call(['rm', '-r', this_path])


def get_timestamps(session_data, n_trials, n_trace_types, meta_fs=None, tiff_counts=None, fudge=0):

    timestamps = {'align': np.full(n_trials, np.nan),
                  # 'abs_align': np.full(n_trials, np.nan),
                  'frame_starts': [[]] * n_trials,
                  'frame_ends': [[]] * n_trials,
                  'frame_midpoints': [[]] * n_trials,
                  'stim': np.full(n_trials, np.nan),
                  'trace': np.full(n_trials, np.nan),
                  'fs': np.full(n_trials, np.nan),
                  'tiffs_per_trial': np.full(n_trials, np.nan),
                  #                   'tiff_starts': np.full(n_trials, np.nan),
                  'first_trial': 0,
                  'last_trial': n_trials,
                  'iti': 2,  # duration after reward to plot
                  'foreperiod': 1  # duration before CS to plot
                  }

    if tiff_counts is not None:
        timestamps['tiff_starts'] = np.cumsum(tiff_counts) - tiff_counts
        continuous = 0
    else:
        continuous = 1

    # housekeeping kludge, because in some protocols the last state of the protocol is called "TrialEnd" and in other
    # protocols the last state is called ITI (in which case there is no TrialEnd state)
    if hasattr(session_data['RawEvents']['Trial'][0].States, 'TrialEnd'):
        last_state_name = 'TrialEnd'
    elif hasattr(session_data['RawEvents']['Trial'][0].States, 'Pause'):
        last_state_name = 'Pause'
    else:
        last_state_name = 'ITI'

    # changed hardware config on 20200717
    if 'exp_date' in session_data:
        field_prefix = 'exp'
    else:
        field_prefix = 'file'
    if int(session_data[field_prefix + '_date']) >= 20200717:
        high = 'BNC1High'
        low = 'BNC1Low'
    else:
        high = 'Wire1High'
        low = 'Wire1Low'

    if continuous:
        # first, we need to get the first TTL and the empirical frame rate from the first FULL trial
        timestamps['first_ttl_time'] = 0
        for i in range(n_trials):
            timestamps['tiffs_per_trial'][i] = 0

            if hasattr(session_data['RawEvents']['Trial'][i].Events, high):
                # first_ttl_midpoint = (session_data['RawEvents']['Trial'][i].Events.Wire1High[0] +
                # 						session_data['RawEvents']['Trial'][i].Events.Wire1Low[0]) / 2

                first_ttl_midpoint = (getattr(session_data['RawEvents']['Trial'][i].Events, high)[0] +
                                      getattr(session_data['RawEvents']['Trial'][i].Events, low)[0]) / 2
                timestamps['first_ttl_time'] = first_ttl_midpoint + session_data['TrialStartTimestamp'][i]

                # compute number of frames occurring between first imaging trial and start of the following trial
                fs = 1. / np.mean(np.diff(getattr(session_data['RawEvents']['Trial'][i].Events, high)))

                timestamps['tiffs_per_trial'][i] = len(getattr(session_data['RawEvents']['Trial'][i].Events, low)) + \
                                                   (session_data['RawEvents']['Trial'][i].States.ITI[-1] + .02 - \
                                                    getattr(session_data['RawEvents']['Trial'][i].Events, low)[-1]) * fs
                break
            if i == n_trials - 1:
                raise_print('No Wire1High/BNC1High events detected. Check the Wire1 connection on the Bpod device.')
    else:
        ttl_counts = np.zeros((n_trials,))
        timestamps['tiffs_per_trial'] = tiff_counts[:n_trials]

    for i in range(n_trials):

        # get timestamps of end of foreperiod/start of odor
        timestamps['align'][i] = session_data['RawEvents']['Trial'][i].States.Foreperiod[-1]

        # calculate stimulus and trace duration from Bpod, rather than assuming it
        timestamps['trace'][i] = session_data['RawEvents']['Trial'][i].States.Trace[-1] - \
                                 session_data['RawEvents']['Trial'][i].States.Trace[0]

        if session_data['TrialTypes'][i] <= n_trace_types:
            # need two different lines for stimulus duration, because stimulus delivery state varies between trials
            stimulus_field = getattr(session_data['RawEvents']['Trial'][i].States,
                                     'Stimulus' + str(session_data['TrialTypes'][i]) + 'Delivery')
            timestamps['stim'][i] = stimulus_field[-1] - stimulus_field[0]
        elif session_data['TrialTypes'][i] > n_trace_types:
            timestamps['stim'][i] = np.nan
        else:
            raise_print('Trial type not recognized.')

        # if continuous, and TTLs were sent on this trial. This will always be true if looped
        # if there were no TTLs sent on that trial, then Wire1High/Low won't exist as fields
        if hasattr(session_data['RawEvents']['Trial'][i].Events, high):

            timestamps['fs'][i] = 1. / np.mean(np.diff(getattr(session_data['RawEvents']['Trial'][i].Events, high)))

            # trial_end_time will be misleading for looped acquisitions, since the microscope may have disarmed by then.
            # but it won't be used in this case
            trial_end_time = getattr(session_data['RawEvents']['Trial'][i].States, last_state_name)[-1]

            # ignore putative first trial, which we computed above. Never true if looped acquisition
            if i < n_trials - 1 and np.isnan(timestamps['tiffs_per_trial'][i]):
                # this will overestimate the tiff starts after we stop imaging, but at that point we won't be looking
                # for TIFFs anyway
                timestamps['tiffs_per_trial'][i] = (session_data['TrialStartTimestamp'][i + 1] -
                                                       session_data['TrialStartTimestamp'][i]) * (
                                                                  timestamps['fs'][i] - fudge)
                # timestamps['tiff_starts'][i + 1] = (session_data['TrialStartTimestamp'][i + 1] -
                #                                     session_data['TrialStartTimestamp'][i]) * float(meta_fs) + \
                #                                    timestamps['tiff_starts'][i]

            trial_framepositions_start = copy(getattr(session_data['RawEvents']['Trial'][i].Events, high))
            trial_framepositions_end = copy(getattr(session_data['RawEvents']['Trial'][i].Events, low))

            if continuous:
                # handle (very common) case where frames are in progress at start/end of trial, so the TTL gets split
                # between consecutive trials. If majority of frame belongs to this trial, add the other part of the TTL
                if trial_framepositions_start[0] > trial_framepositions_end[0]:
                    if trial_framepositions_end[0] > 1. / (2 * timestamps['fs'][i]):
                        trial_framepositions_start = np.insert(trial_framepositions_start, 0,
                                                               trial_framepositions_start[0] - 1. / timestamps['fs'][i])
                    else:
                        trial_framepositions_end = np.delete(trial_framepositions_end, 0)
                if trial_framepositions_start[-1] > trial_framepositions_end[-1]:
                    if trial_end_time - trial_framepositions_start[-1] > 1. / (2 * timestamps['fs'][i]):
                        trial_framepositions_end = np.insert(trial_framepositions_end, len(trial_framepositions_end),
                                                             trial_framepositions_end[-1] + 1. / timestamps['fs'][i])
                    else:
                        trial_framepositions_start = np.delete(trial_framepositions_start, -1)

                # before the covid shutdown, I was manually starting scanimage after starting bpod
                # after the shutdown, I changed the configuration to allow start and stop triggers, so first and last trial
                # shouldn't need changing. Switch to BNC1 was after this, so leave it as Wire1High/Low.
                if int(session_data[field_prefix + '_date']) < 20200608:

                    # the first trial is the one where TTLs have been going the whole time. So if there's a gap of more than
                    # 1.1 fs, don't use that trial
                    if session_data['RawEvents']['Trial'][i].Events.Wire1High[0] < 1.1 / timestamps['fs'][i] and \
                            timestamps['first_trial'] == 0:
                        timestamps['first_trial'] = i

                    # similarly, last trial (inclusive) to use is the one before the trial where TTLs stop before the end of
                    # the ITI. But we want the last trial *exclusive*, (that is, the number of trials) so don't subtract 1
                    elif session_data['RawEvents']['Trial'][i].Events.Wire1Low[-1] < trial_end_time - (
                            1.1 / timestamps['fs'][i]):
                        timestamps['last_trial'] = i

                # timestamps['abs_align'][i] = session_data['TrialStartTimestamp'][i] + timestamps['align'][i] - \
                #                              timestamps['first_ttl_time']

                timestamps['tiff_starts'] = np.insert(np.cumsum(timestamps['tiffs_per_trial'])[:-1], 0, 0)
            else:
                # on final trial of looped acquisition, stopping mid-frame can mean Bpod never receives a TTL low
                if i == n_trials - 1:
                    trial_framepositions_start = trial_framepositions_start[:len(trial_framepositions_end)]
                # Wire 1 is the TTL input from the microscope back to Bpod to indicate the frame duration and frame number
                # for some reason, TTLs are sometimes sent back to Bpod after the ITI ends, even though frames are not saved.
                # So, manually remove those extra "frames" now. Take up to the last frame completed before the end of the ITI
                selector = getattr(session_data['RawEvents']['Trial'][i].Events, low) <= \
                           session_data['RawEvents']['Trial'][i].States.ITI[-1]
                trial_framepositions_start = trial_framepositions_start[selector]
                trial_framepositions_end = trial_framepositions_end[selector]
                ttl_counts[i] = len(trial_framepositions_start)

            timestamps['frame_starts'][i] = trial_framepositions_start
            timestamps['frame_ends'][i] = trial_framepositions_end
            timestamps['frame_midpoints'][i] = np.add(trial_framepositions_start, trial_framepositions_end) / 2

        else:
            # generally, this will be in the continuous case, without external triggering
            if timestamps['first_trial'] == i:  # if it for some reason started late?
                timestamps['first_trial'] += 1
            # case where ScanImage is stopped during dead time and acquisition is continuous
            elif timestamps['last_trial'] == n_trials and timestamps['first_trial'] != 0:
                timestamps['last_trial'] = i

    if continuous:
        s_per_trial_bpod = np.diff(session_data['TrialStartTimestamp'])
        #         s_per_trial_tiffs = np.diff(timestamps['tiff_starts']) / timestamps['fs'][:-1]
        s_per_trial_tiffs = timestamps['tiffs_per_trial'] / timestamps['fs']
        # ignore first trial b/c that's when scanimage is triggered, and there is some lag that has already been accounted for
        error = s_per_trial_tiffs[1:-1] - s_per_trial_bpod[1:]
        if np.amax(np.abs(error)) > 1 / np.nanmean(timestamps['fs']):
            raise_print('Misaligned')

    if not continuous and not np.all(np.abs(ttl_counts[:n_trials] - tiff_counts[:n_trials]) <= 1):
        raise_print('Bpod and ScanImage do not agree on number of frames')

    timestamps = validate_timestamps(timestamps)
    return timestamps

def count_scanimage_tiffs(video_dir, i_tiffs=None):
    tiff_lens = []
    filenames = []
    for filename in os.listdir(video_dir):
        if filename.endswith('.tif'):
            filenames.append(filename)
    # os.listdir returns files in random order, which screws things up if you're not careful!

    if i_tiffs is None:
        i_tiffs = np.arange(len(filenames))

    for i_tiff in i_tiffs:
        with ScanImageTiffReader.ScanImageTiffReader(os.path.join(video_dir, sorted(filenames)[i_tiff])) as reader:
            nframes = reader.shape()[0]
            tiff_lens.append(nframes)

            if i_tiff == 0:
                meta = dict(x.split(' = ') for x in reader.description(0).split('\n', maxsplit=150))
            reader.close()

    return np.array(tiff_lens), meta['scanimage.SI4.scanFrameRate']


def get_temp_dir(db_dict, video_dir):
    if on_cluster() and not db_dict['continuous']:
        # rsync data over to scratch first, because will be loading in a bunch of tiffs (not strictly necessary, but do it
        # because we don't want to 100% trust Bpod
        scratch_data_folder = os.path.join(os.environ['SCRATCH'], 'uchida_lab', 'alowet', 'imaging', db_dict['name'])
        check_dir(scratch_data_folder)
        subprocess.call(['rsync', '-avx', '--progress', video_dir, scratch_data_folder])
        temp_dir = os.path.join(scratch_data_folder, db_dict['file_date_id'])
    else:
        temp_dir = video_dir
    return temp_dir


def get_pdf_dir(db_dict, video_dir):
    if on_cluster():
        scratch_data_folder = os.path.join(os.environ['SCRATCH'], 'uchida_lab', 'alowet', 'imaging', db_dict['name'])
        temp_dir = os.path.join(scratch_data_folder, db_dict['file_date_id'])
        check_dir(temp_dir)
    else:
        temp_dir = video_dir
    print('PDF temp dir: ', temp_dir)
    return temp_dir


