from suite2p.run_s2p import default_ops

def get_kludge(data, code):
    # for sessions where I got the ScanImage settings wrong
    if data['file_date'] == '20191221':
        code = -1
    return code

def load_ops(code, save_path, data, meta):

    # load ops file. This can and should be the same for all recording sessions
    ops = default_ops()
    ops['save_path0'] = save_path  # stores results, defaults to first item in data_path
    ops['delete_bin'] = False  # whether to delete binary file after processing
    ops['nplanes'] = 1  # each tiff has these many planes in sequence

    ops['smooth_sigma_time'] = 0  # gaussian smoothing in time
    ops['sparse_mode'] = False  # whether or not to run sparse_mode
    ops['diameter'] = 20  # 22 when 1.6x zoom  # if not sparse_mode, use diameter for filtering and extracting
    ops['threshold_scaling'] = 1.0  # 0.8  # adjust the automatically determined threshold by this scalar multiplier
    ops['high_pass'] = 75  # running mean subtraction with window of size 'high_pass' (use low values for 1P)
    ops['inner_neuropil_radius'] = 2  # number of pixels to keep between ROI and neuropil donut
    ops['num_workers'] = 0  # 0 to select num_cores, -1 to disable parallelism, N to enforce value

    # added on 6/19/20
    ops['neucoeff'] = 0.58

    # added on 6/6/22
    ops['classifier_path'] = '/n/home06/alowet/grin_classifier.npy'

    if code == 1:
        # Channel 1: GRAB-DA2m
        # Channel 2: jrGECO1a
        ops['nchannels'] = 2  # each tiff has these many channels per plane
        ops['functional_chan'] = 2
        ops['align_by_chan'] = 2
        ops['tau'] = 0.7  # To match GCaMP6f recommendation. Dana et al., 2016, Fig. 2E.
        ops['chan_names'] = ['DA', 'Ca']
    elif code == 2:
        # Channel 1: GRAB-DA2m
        # Channel 2: Not acquired
        ops['nchannels'] = 1
        ops['functional_chan'] = 1
        ops['tau'] = 3.7  # Sun et al., 2018, Fig. 2F
        ops['chan_names'] = ['DA']
    elif code == 3:
        # Channel 1: GCaMP7f
        # Channel 2: Not acquired
        ops['nchannels'] = 1
        ops['functional_chan'] = 1
        ops['tau'] = 0.7  # Official Suite2P recommendation for GCaMP6f. Dana et al., 2019, Fig. 2F, and Chen et al., 2013, Fig. 1f
        ops['chan_names'] = ['Ca']
    elif code == 4:
        # Channel 1: GCaMP7f
        # Channel 2: GRAB-rDA1m
        ops['nchannels'] = 2  # each tiff has these many channels per plane
        ops['functional_chan'] = 1
        ops['tau'] = 0.7  # To match GCaMP6f recommendation. Dana et al., 2016, Fig. 2E.
        ops['chan_names'] = ['Ca', 'DA']
    elif code == 5:
        # Channel 1: GCaMP7s or 6s
        # Channel 2: Not acquired
        ops['nchannels'] = 1
        ops['functional_chan'] = 1
        ops['tau'] = 2.0  # To match GCaMP7s. Dana et al., 2019, Fig. 2F, and Chen et al., 2013, Fig. 1f. See also https://github.com/cortex-lab/Suite2P/issues/159
        ops['chan_names'] = ['Ca']
    elif code == 6:
        # Channel 1: jrGECO1a
        # Channel 2: Not acquired
        ops['nchannels'] = 1  # each tiff has these many channels per plane
        ops['functional_chan'] = 1
        ops['tau'] = 0.7  # To match GCaMP6f recommendation. Dana et al., 2016, Fig. 2E.
        ops['chan_names'] = ['Ca']
    elif code == 7:
        # Channel 1: GRAB-DA2/3m
        # Channel 2: tdTomato (Cre-dependent, transgenic)
        ops['nchannels'] = 2
        ops['functional_chan'] = 1
        ops['tau'] = 3.7  # Sun et al., 2018, Fig. 2F
        ops['chan_names'] = ['DA', 'tdT']
        ops['align_by_chan'] = 2
    elif code == -1:
        # Negative codes are for kludges, when I messed up something with the ScanImage acquisition
        # Channel 1: GRAB-DA2m
        # Channel 2: Empty, but acquired
        ops['nchannels'] = 2
        ops['functional_chan'] = 1
        ops['tau'] = 3.7
        ops['chan_names'] = ['DA', 'Ca']
    else:
        print('Code ' + code + ' not recognized. Check code handling in s2p_db_utils.load_ops')
        raise Exception('Code ' + code + ' not recognized. Check code handling in s2p_db_utils.load_ops')

    if data['file_date_id'].endswith('red'):
        # imaging at 1040 nm, so that functional channel is chan 2. Both channels acquired
        ops['nchannels'] = 2  # each tiff has these many channels per plane
        ops['functional_chan'] = 2  # DA is functional chan here
        ops['align_by_chan'] = 2  # don't expect much fluo in green channel
        ops['tau'] = 3.7  # A guess for rDA1m, based off of DA1m
        ops['chan_names'] = ['Ca', 'DA']

    # use the scanimage metadata to set some suite2p params
    ops['fs'] = float(meta['scanimage.SI4.scanFrameRate'])
    if 'bidirectional' in meta['scanimage.SI4.scanMode']:
        ops['do_bidiphase'] = True

    n_chan = len(meta['scanimage.SI4.channelsSave'].split(';'))
    if n_chan != ops['nchannels']:
        # usually, this occurs when saving two channels, even though only one has a fluorophore. Just use the active
        # channel and print a warning
        print('{} channel(s) were saved, but there are {} active for this mouse. Proceeding with {} channels.'.format(
            n_chan, ops['nchannels'], n_chan))
        ops['nchannels'] = n_chan
    if n_chan == 1:
        ops['functional_chan'] = 1
        ops['align_by_chan'] = 1

    return ops




