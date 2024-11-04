import numpy as np
from basis import *

def parse_group_from_feature_names(feature_names):
    '''
    Parse feature_names into groups using hand-crafted rules

    Input parameters::
    feature_names: List of feature names. In this example, expanded features must contain bumpX in the name

    Returns::
    group_size: list of number of features in each group
    group_name: name of each group
    group_ind: group index of each feature in feature_names, ndarray of size (len(feature_names),)
    '''

    # Find expanded features and their number of sub-features:
    group_size = list()
    group_name = list()
    group_ind = list()
    for name in feature_names:
        if 'bump' not in name and 'mot_svd' not in name and 'filt' not in name:
            # Non-bump expanded feature:
            group_size.append(1)
            group_name.append(name)

        elif 'bump0' in name or 'filt0' in name:
            # First bump of a bump-expanded feature:
            group_size.append(1)
            group_name.append(name[:-5])

        elif 'mot_svd0' in name:
            group_size.append(1)
            group_name.append('mot_svd')

        else:
            # Subsequent time shifts and bumps
            group_size[-1] += 1

            # Create group index for each feature
    for i_group, this_size in enumerate(group_size):
        group_ind += [i_group] * this_size

    return group_size, group_name, np.array(group_ind)


def get_filts(dt):
    # Licks, running, reward presence, reward magnitude deliered, and rpe are modeled as discrete events, convovled
    # with a set of filters
    n_filts = 5
    last_hpeak = 2
    # dt = neural.dt
    # 2-vector containg [1st_peak  last_peak], the peak location of first and last raised cosine basis vectors
    hpeaks = np.array([dt, last_hpeak])
    b = last_hpeak / 5  # offset for nonlinear stretching of x axis:  y = log(x+b) (larger b -> more nearly linear stretching)

    # Generate basis of raised cosines
    yrnge = nlin(hpeaks + b)  # nonlinearly transformed first & last bumps
    dbump = float(np.diff(yrnge) / (n_filts - 1))  # spacing between cosine bump peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + dbump / 2, dbump)  # centers (peak locations) for basis vectors

    # Make basis
    mxt = invnl(yrnge[1] + 2 * dbump) - b  # maximum time bin
    iht = np.arange(dt, mxt + dt, dt)[:, np.newaxis]
    nt = len(iht)  # number of points in iht

    ihbasis = ff(np.matlib.repmat(nlin(iht + b), 1, n_filts), np.matlib.repmat(ctrs, nt, 1), dbump)
    # plt.plot(np.arange(0, nt * dt, dt), ihbasis)
    # plt.title('Filters')
    filts = np.zeros((nt * 2, n_filts))
    filts[nt:, :] = ihbasis
    filt_time = np.arange(-nt * dt, nt * dt, dt)
    # plt.figure()
    # plt.plot(filt_time, filts)
    # plt.title('Filters (neural activity -> variable)')
    pred_filts = np.zeros((nt * 2, n_filts))
    pred_filts[:nt, :] = ihbasis[::-1, :]
    # plt.figure()
    # plt.plot(filt_time, pred_filts)
    # plt.title('Predictive filters (variable -> neural activity)')
    return filts, pred_filts, filt_time
