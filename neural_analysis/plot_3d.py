import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.special import loggamma
from scipy.special import expit
from scipy.optimize import minimize
from scipy import stats
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
import cmocean
import sys
sys.path.append('../utils')
from plotting import hide_spines, add_cbar

def betaLogLikelihood(params, y, X):
    b = np.array(params[0:-1])  # the beta parameters of the regression model
    phi = params[-1]  # the phi parameter
    mu = expit(np.dot(X, b))

    eps = 1e-6  # used for safety of the gamma and log functions avoiding inf
    res = - np.sum(loggamma(phi + eps)  # the log likelihood
                   - loggamma(mu * phi + eps)
                   - loggamma((1 - mu) * phi + eps)
                   + (mu * phi - 1) * np.log(y + eps)
                   + ((1 - mu) * phi - 1) * np.log(1 - y + eps))

    return res



def spatial_regression_mouse_3d(ap, ml, dv, names, vals):

    X, Y, Z = np.mgrid[np.percentile(ml, 2.5):np.percentile(ml, 97.5):7j,
              np.percentile(ap, 2.5):np.percentile(ap, 97.5):7j,
              np.percentile(dv, 2.5):-4:7j]
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    predictor_mat = np.stack([X, Y, Z, X * Y, X * Z, Y * Z, X * Y * Z], axis=1)

    mice = np.unique(names)
    scores = np.zeros(len(mice))
    y_smooths = np.zeros((len(mice), *predictor_mat.shape))
    y_ests = []

    for i_mouse, mouse_name in enumerate(mice):
        mi = names == mouse_name
        reg_mat = np.stack([ml[mi], ap[mi], dv[mi], ml[mi] * ap[mi], ml[mi] * dv[mi], ap[mi] * dv[mi], ml[mi] * ap[mi] * dv[mi]], axis=1)
        # X, Y, Z = np.mgrid[np.min(ml):np.max(ml):3j, np.min(ap):np.max(ap):2j, np.min(dv):np.max(dv):7j]

        # scoring = 'R^2'
        alphas = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
        LR = RidgeCV(alphas=alphas, cv=None).fit(reg_mat, vals[mi])  # leave-one-out CV

        y_ests.append(LR.predict(reg_mat))
        y_smooths[i_mouse] = LR.predict(predictor_mat)
        scores[i_mouse] = LR.score(reg_mat, vals)
        # title = r'CV ${} = {:.3f}$'.format(scoring, score)

    coords = [X, Y, Z]
    return scores, y_ests, y_smooths, coords


def spatial_regression_3d(ap, ml, dv, vals):

    reg_mat = np.stack([ml, ap, dv, ml * ap, ml * dv, ap * dv, ml * ap * dv], axis=1)
    # X, Y, Z = np.mgrid[np.min(ml):np.max(ml):3j, np.min(ap):np.max(ap):2j, np.min(dv):np.max(dv):7j]
    X, Y, Z = np.mgrid[np.percentile(ml, 2.5):np.percentile(ml, 97.5):7j,
                       np.percentile(ap, 2.5):np.percentile(ap, 97.5):7j,
                       np.percentile(dv, 2.5):-4:7j]
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    predictor_mat = np.stack([X, Y, Z, X * Y, X * Z, Y * Z, X * Y * Z], axis=1)

    if np.nanmin(vals) >= 0 and np.nanmax(vals) <= 1 and vals.dtype == float:  # beta regression, coded by hand
        # initial parameters for optimization
        phi = 1
        b0 = 1
        x0 = np.array([b0, b0, b0, b0, b0, b0, b0, phi])

        res = minimize(betaLogLikelihood, x0=x0, args=(vals, reg_mat), bounds=[(None, None),
                                                                               (None, None),
                                                                               (None, None),
                                                                               (None, None),
                                                                               (None, None),
                                                                               (None, None),
                                                                               (None, None),
                                                                               (0, None)])

        b = np.array(res.x[0:reg_mat.shape[1]])  # optimal regression parameters
        y_est = expit(np.dot(reg_mat, b))  # predictions
        y_smooth = expit(np.dot(predictor_mat, b))  # predictions for meshgrid
        score = None  # not yet implemented
        title = 'Beta regression (not CV)'

    else:
        if vals.dtype == int:  # multinomial logistic regression
            scoring = 'balanced_accuracy'
            LR = LogisticRegressionCV(cv=10, multi_class='multinomial', scoring='balanced_accuracy',
                                      max_iter=10000).fit(reg_mat, vals)
        else:  # linear regression
            scoring = 'R^2'
            alphas = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
            LR = RidgeCV(alphas=alphas, cv=None).fit(reg_mat, vals)  # leave-one-out CV

        y_est = LR.predict(reg_mat)
        y_smooth = LR.predict(predictor_mat)
        score = LR.score(reg_mat, vals)
        title = r'CV ${} = {:.3f}$'.format(scoring, score)

    coords = [X, Y, Z]
    return score, y_est, y_smooth, title, coords


def brain_3d(neuron_info, cval, inds, label, cinds=None, jit=0, vmin=None, vmax=None):

    inds = np.logical_and(inds, ~np.isnan(cval))

    # plot cval as a function of coordinate
    ap = np.array(neuron_info['aps'])[inds]
    ml = np.abs(np.array(neuron_info['mls'])[inds])
    dv = np.array(neuron_info['depths'])[inds]
    names = neuron_info['names'][inds]

    if cinds is None:
        cinds = np.ones(len(cval), dtype=bool)
    elif type(cinds) == str and cinds == 'same':
        cinds = inds

    y_true = cval[cinds]
    # this (old) version pools across mice indiscriminately
    score, y_est, y_smooth, title, [X, Y, Z] = spatial_regression_3d(ap, ml, dv, y_true)
    # spatial_regression_mouse_3d(ap, ml, dv, names, y_true)

    plt.figure()
    plt.scatter(y_true, y_est)
    r, p = stats.pearsonr(y_true, y_est)
    plt.title(r"Pearson's $r={:.3f}, p={:.3f}$".format(r, p))
    # unity = np.arange(np.min(y_true), np.max(y_true), .1)
    # plt.plot(unity, unity, 'k--')
    plt.xlabel('Actual {}'.format(label))
    plt.ylabel('Predicted {}'.format(label))
    hide_spines()

    if np.nanmin(y_true) < 0:
        imin = np.nanmin(y_true)
        cmap = cmocean.tools.crop(cmocean.cm.balance_r, np.nanmin(y_true), np.nanmax(y_true), pivot=0)
        cscale = [[cmap._segmentdata['red'][x][0], 'rgb({}, {}, {})'.format(
            cmap._segmentdata['red'][x][1], cmap._segmentdata['green'][x][1], cmap._segmentdata['blue'][x][1])]
                  for x in range(len(cmap._segmentdata['red']))]
    else:
        imin = 0
        cmap = plt.cm.plasma
        cscale = 'plasma'
    imax = max(np.nanmax(y_true), 1)

    plt.figure()
    ax = plt.axes(projection='3d')
    im = ax.scatter3D(ml + np.random.normal(scale=jit, size=len(ap)),
                      ap + np.random.normal(scale=jit, size=len(ap)),
                      dv, c=y_true, cmap=cmap, vmin=vmin, vmax=vmax)

    add_cbar(plt.gcf(), im, label, d3=True)
    ax.set_xlabel('ML (mm)', labelpad=10)
    ax.set_ylabel('AP (mm)', labelpad=10)
    ax.set_zlabel('DV (mm)', labelpad=10)
    ax.set_title(title)

    fig = go.Figure(data=go.Volume(
        x=X, y=Y, z=Z, value=y_smooth, opacity=0.1, isomin=imin, isomax=imax,
        surface_count=5000, colorscale=cscale))
    fig.update_layout(scene=dict(xaxis_title='ML (mm)', yaxis_title='AP (mm)', zaxis_title='DV (mm)'),
        autosize=False, width=500, height=500)
    fig.update_traces(colorbar_thickness=10)
    fig.show()

    return score, y_smooth, y_true, [X, Y, Z], [ap, ml, dv]


def shuffle_locations(dist_neurons, modulation_index, neuron_info, to_shuff='mouse', n_shuff=1000):

    shuff_reg_scores = np.zeros(n_shuff)
    rng = np.random.default_rng(seed=1)

    inds = np.logical_and(dist_neurons, ~np.isnan(modulation_index))

    # get coords of these neurons
    ap = np.array(neuron_info['aps'])[inds]
    ml = np.abs(np.array(neuron_info['mls'])[inds])
    dv = np.array(neuron_info['depths'])[inds]

    shuff_modlu_index = np.full(dist_neurons.shape[0], np.nan)  # total_cells
    mice = np.unique(neuron_info['names'])

    for i_shuff in range(n_shuff):

        if to_shuff == 'mouse':
            # randomly shuffle which scores go where, within mouse
            for mouse_name in mice:
                mouse_inds = np.logical_and(inds, neuron_info['names'] == mouse_name)
                shuff_modlu_index[mouse_inds] = rng.permutation(modulation_index[mouse_inds])
        else:  # randomly shuffle which scores go where, across mice
            shuff_modlu_index[inds] = rng.permutation(modulation_index[inds])

        shuff_reg_scores[i_shuff], _, _, _, _ = spatial_regression_3d(ap, ml, dv, shuff_modlu_index[inds])

    return shuff_reg_scores


def plot_shuff_dist(shuff_reg_scores, score):
    plt.figure(figsize=(3, 2))
    plt.hist(shuff_reg_scores, density=False, edgecolor='k', facecolor=[.75,.75,.75])
    plt.vlines(x=score, ymin=0, ymax=plt.ylim()[1], ls='--', color='k')
    plt.ylabel('Number of shuffled\nobservations')
    plt.xlabel('Cross-validated $R^2$')
    bootp = np.mean(score <= shuff_reg_scores)
    plt.title('$p = {:0.3f}$'.format(bootp))
    hide_spines()
    return bootp

