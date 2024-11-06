import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.optimize import root, minimize
from datetime import datetime
import os
from scipy import stats, odr, linalg, spatial
import cmocean
import fitz
import sys
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import confusion_matrix, silhouette_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import itertools
import time
import pandas as pd
from pathlib import Path
import json

from streams import DataStream

sys.path.append('../utils')
from db import get_db_info
from plotting import add_cbar_and_vlines, set_share_axes, hide_spines, add_cbar, plot_confusion, plot_box
from protocols import load_params
from paths import raise_print
from matio import loadmat
sys.path.append('../behavior_analysis')
from traceUtils import setUpLickingTrace


def expectile_loss_fn(expectiles, taus, samples):
    """Expectile loss function, corresponds to distributional TD model.
	 Returns a single scalar, the mean squared error"""
    # distributional TD model: delta_t = (r + \gamma V*) - V_i
    # expectile loss: delta = sample - expectile
    delta = (samples[None, :] - expectiles[:, None])

    # distributional TD model: alpha^+ delta if delta > 0, alpha^- delta otherwise
    # expectile loss: |taus - I_{delta <= 0}| * delta^2

    # Note: When used to decode we take the gradient of this loss,
    # and then evaluate the mean-squared gradient. That is because *samples* must
    # trade-off errors with all expectiles to zero out the gradient of the
    # expectile loss.
    indic = np.array(delta <= 0., dtype=np.float32)
    grad = -0.5 * np.abs(taus[:, None] - indic) * delta
    return np.mean(np.square(np.mean(grad, axis=-1)))


def expectile_grad_loss(expectiles, taus, dist):
    # returns a vector, one value for each expectile
    delta = dist[np.newaxis, :] - expectiles[:, np.newaxis]
    indic = np.array(delta <= 0., dtype=np.float32)
    grad = -2. * np.abs(taus[:, np.newaxis] - indic) * delta
    return np.mean(grad, axis=1)


def run_decoding(reversal_points, taus, minv=0., maxv=1., method=None,
                 max_samples=1000, max_epochs=3, N=100):
    """Run decoding given reversal points and asymmetries (taus)."""

    ind = list(np.argsort(reversal_points))
    points = reversal_points[ind]
    tau = taus[ind]

    # Robustified optimization to infer distribution
    # Generate max_epochs sets of samples,
    # each starting the optimization at the best of max_samples initial points.
    sampled_dist = []
    for _ in range(max_epochs):
        # Randomly search for good initial conditions
        # This significantly improves the minima found
        samples = np.random.uniform(minv, maxv, size=(max_samples, N))
        fvalues = np.array([expectile_loss_fn(points, tau, x0) for x0 in samples])

        # Perform loss minimizing on expectile loss (w.r.t samples)
        x0 = np.array(sorted(samples[fvalues.argmin()]))
        fn_to_minimize = lambda x: expectile_loss_fn(points, tau, x)
        result = minimize(fn_to_minimize, method=method, bounds=[(minv, maxv) for _ in x0], x0=x0)['x']

        sampled_dist.extend(result.tolist())

    return sampled_dist, expectile_loss_fn(points, tau, np.array(sampled_dist))


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def linear_func(B, x):
    """
	From https://docs.scipy.org/doc/scipy/reference/odr.html
	Linear function y = m*x + b
	"""
    return B[0] * x + B[1]


def infer_dist(expectiles=None, taus=None, dist=None):
    """
	Given kappa and two of the following three values (reversal_points, taus, and dist), we can always infer the third
	:param reversal_points: vector of expectiles or Huber quantiles
	:param taus: vector, with values between 0 and 1
	:param dist: the distribution for which we want to compute Huber quantiles
	:param kappa: parameter for Huber quantiles
	"""
    # infer expectiles
    if expectiles is None:
        fn_to_solve = lambda x: expectile_grad_loss(x, taus, dist)
        taus[taus < 0.] = 0.
        taus[taus > 1.] = 1.
        sol = root(fn_to_solve, x0=np.quantile(dist, taus), method='lm')

    # infer taus
    elif taus is None:
        fn_to_solve = lambda x: expectile_grad_loss(expectiles, x, dist)
        sol = root(fn_to_solve, x0=np.linspace(0.01, 0.99, len(expectiles)), method='lm')

    # impute distribution
    elif dist is None:
        fn_to_solve = lambda x: expectile_grad_loss(expectiles, taus, x)
        sol = root(fn_to_solve, x0=expectiles, method='lm', options={'maxiter': 100000})

    check_convergence(sol)

    # return the optimized value
    return sol['x']


def plot_imputation(empirical_dist, imputed_dist, ax, bw=.05):
    sns.kdeplot(empirical_dist, bw=bw, color='k', lw=3., shade=True, label="Empirical", ax=ax, legend=False)
    sns.rugplot(empirical_dist, color='k', ax=ax)
    sns.kdeplot(imputed_dist, bw=bw, color=plt.cm.plasma(0), lw=3., shade=True, label="Decoded", ax=ax, legend=False)
    sns.rugplot(imputed_dist, color=plt.cm.plasma(0), ax=ax)


def check_convergence(sol):
    # make sure optimization has converged
    if not sol['success']:
        raise_print(sol['message'])


def heatmap_tt(n_trial_types, trial_type_names, pcolor_time, timecourse, colors, label='', cells_per_sess=None,
               cmap=plt.cm.magma):
    """
    Make one big heatmap of each level, where levels are neurons or components from PCA, NMF, etc.
    :param n_trial_types: Usually 6
    :param pcolor_time:
    :param timecourse: 3D-array, n_trial_types x n_levels x len(std_time)
    :param colors: list of len = n_trial_types
    :return: fig, axs, im
    """
    std_time = (pcolor_time[:-1] + pcolor_time[1:]) / 2

    # make one big heatmap
    grand_mean = np.nanmean(timecourse, axis=1)
    grand_sem = stats.sem(timecourse, axis=1, nan_policy='omit')

    prc_range = np.nanpercentile(timecourse, [2.5, 97.5], axis=None)
    if label == 'Activity' or prc_range[0] > 0:  # for NMF
        this_cmap = plt.cm.gray_r
    else:
        this_cmap = cmocean.tools.crop(cmocean.cm.balance, prc_range[0], prc_range[1], pivot=0)
    total_comps = timecourse.shape[1]

    fig, axs = plt.subplots(2, n_trial_types, figsize=(15, 5), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    [set_share_axes(axs[i, :], sharey=True) for i in range(axs.shape[0])]

    for i in range(n_trial_types):
        ax = axs[0, i]
        im = ax.pcolormesh(pcolor_time, np.arange(total_comps + 1), timecourse[i, :, :], vmin=prc_range[0],
                           vmax=prc_range[1], cmap=this_cmap)
        if cells_per_sess is not None:
            ax.hlines(np.cumsum(cells_per_sess)[:-1], -1, 5)
        ax.set_title(trial_type_names[i])
        ax.set_ylim(total_comps, 0)

        ax = axs[1, i]
        ax.plot(std_time, grand_mean[i], c=colors[i], lw=2)
        ax.fill_between(std_time, grand_mean[i] + grand_sem[i], grand_mean[i] - grand_sem[i], color=colors[i], alpha=.2)
        if i == n_trial_types // 2:
            ax.set_xlabel('Time from CS (s)')

    return fig, axs, im


def heatmap_2d(n_trial_types, rew_types, combo_types, trial_type_names, pcolor_time, timecourse, colors, paths,
               label='', cmap=plt.cm.magma):
    """
    For when I want to break up heatmaps by CS and reward amount. Otherwise, same as heatmap_tt
    """
    std_time = (pcolor_time[:-1] + pcolor_time[1:]) / 2

    # make one big heatmap
    grand_mean = np.nanmean(timecourse, axis=1)
    grand_sem = stats.sem(timecourse, axis=1, nan_policy='omit')

    prc_range = np.nanpercentile(timecourse, [2.5, 97.5], axis=None)
    if label == 'Activity' or prc_range[0] > 0:  # for NMF
        this_cmap = plt.cm.gray_r
    else:
        this_cmap = cmocean.tools.crop(cmocean.cm.balance, prc_range[0], prc_range[1], pivot=0)
    total_comps = timecourse.shape[1]

    n_rew_types = len(rew_types)
    fig, axs = plt.subplots(n_rew_types * 2, n_trial_types, figsize=(3 * n_trial_types, 5 * n_rew_types),
                            gridspec_kw={'height_ratios': [4, 1] * n_rew_types}, sharex=True)
    [set_share_axes(axs[i::2, :], sharey=True) for i in range(2)]

    i_color = 0

    for i in range(n_trial_types):
        for j, rew in enumerate(rew_types):
            ax = axs[j * 2, i]
            combo_type = rew + i * (1 + max(rew_types))
            if combo_type in combo_types:
                combo_ind = list(combo_types).index(combo_type)
            else:
                ax.remove()
                axs[j * 2 + 1, i].remove()
                continue
            im = ax.pcolormesh(pcolor_time, np.arange(total_comps + 1), timecourse[combo_ind, :, :], vmin=prc_range[0],
                               vmax=prc_range[1], cmap=this_cmap)
            ax.set_title(trial_type_names[i] + ' r={}'.format(rew))
            ax.set_ylim(total_comps, 0)
            if i == 0:
                ax.set_ylabel('Neuron #')

            ax = axs[j * 2 + 1, i]
            ax.plot(std_time, grand_mean[i_color], c=colors[i_color], lw=2)
            ax.fill_between(std_time, grand_mean[i_color] + grand_sem[i_color], grand_mean[i_color] - grand_sem[i_color],
                            color=colors[i_color], alpha=.2)
            if i == n_trial_types // 2 and j == n_rew_types - 1:
                ax.set_xlabel('Time from CS (s)')
            if i == 0:
                ax.set_ylabel('Grand mean {}'.format(label))

            i_color += 1

    plt.tight_layout()
    add_cbar_and_vlines(fig, im, 'Mean {}'.format(label), 3.)

    fname = datetime.today().strftime('%Y%m%d') + '_combo.png'
    [plt.savefig(os.path.join(plot_root, 'pooled', fname), bbox_inches='tight', dpi=300) for plot_root in
     paths['neural_fig_roots']]


def label_heatmap(n_trial_types, trial_type_names, pcolor_time, timecourse, colors, total_cells, paths, suffix, label,
                  cells_per_sess=None, yscl=None):
    fig, axs, im = heatmap_tt(n_trial_types, trial_type_names, pcolor_time, timecourse, colors, label, cells_per_sess)
    axs[0, 0].set_ylabel('Neuron #')
    axs[1, 0].set_ylabel('Grand mean {}'.format(label))
    if yscl is not None:
        axs[1, 0].set_ylim(*yscl)
        # for ax in axs[1, :]:
        #     ax.set_ylim(*yscl)
    add_cbar_and_vlines(fig, im, 'Mean {}'.format(label), 3.)
    fname = datetime.today().strftime('%Y%m%d') + '_' + str(total_cells) + suffix
    plt.savefig(os.path.join(paths['neural_fig_roots'][0], fname + '.png'), bbox_inches='tight', dpi=600)
    # plt.savefig(os.path.join(paths['neural_fig_roots'][0]', fname + '.pdf'), bbox_inches='tight')


def get_activity_label(activity):
    """
	Damn my inconsistent naming conventions!
	"""
    if activity == 'firing':
        label = 'FR (std)'
    elif activity == 'F':
        label = 'Fluorescence (std)'
    elif activity == 'dFF':
        label = r'$\Delta F/F$'
    elif activity == 'spks':
        label = 'Activity'
    elif activity == 'spks_smooth':
        label = 'Activity'
    return label


def get_pdf_label(activity):
    """
	Damn my inconsistent naming conventions!
	"""
    if activity == 'F':
        label = 'zF'
    elif activity == 'spks':
        label = 'deconv'
    elif activity == 'firing':
        label = 'spikes'
    else:
        label = activity
    return label


def all_axs_off(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.set_yticks([])
    ax.set_xticks([])


def find_trial_rew_correls(activity, stat_vec):
    """
	:param activity:  n_cells x n_trials x n_steps.
	:param stat_vec: n_trials, the mean reward (or quantile/expectile/higher order statistic) expected in each trial
	:return: corr: n_cells x n_steps x 5 with scipy.linregress of stat_vec on neural activity in each time step
	"""
    n_cells = activity.shape[0]
    n_steps = activity.shape[2]

    assert activity.shape[1] == stat_vec.shape[0]  # both should be n_trials

    corr = np.zeros((n_cells, n_steps, 5))
    for i_cell in range(n_cells):
        for i_step in range(n_steps):
            corr[i_cell, i_step, :] = stats.linregress(stat_vec, activity[i_cell, :, i_step])
    return corr


def find_rew_correls(grand_mean, mean_rews):
    """
	:param grand_mean:  n_cells x n_comp_periods x n_trial_types.
	:param mean_rews: n_trial_types vector of mean rewards (expected or received)
	:return: rew_correls: n_cells x n_comp_periods x 5 with scipy.linregress of mean reward size on neural activity
	"""
    n_cells = grand_mean.shape[0]
    n_comp_periods = grand_mean.shape[1]
    n_tt = len(mean_rews)

    # EFFICIENTLY COMPUTE LINREGRESS using only a single loop, rather than a nested loop
    n = n_cells * n_comp_periods
    fgrand = np.reshape(grand_mean, (n, n_tt))
    rew_correls = np.zeros((n, 5))  # n_cells x n_comp_periods x n_statistics to store
    for i in range(n):
        rew_correls[i] = stats.linregress(fgrand[i], mean_rews)
    rew_correls = np.reshape(rew_correls, (n_cells, n_comp_periods, 5))  # back to 3D
    return rew_correls


# for i_cell in range(n_cells):
#     for i_period in range(n_comp_periods):
# tt_period_mean = []
# tt_period_mean = [X_means[i_cell, tt_inds, i_period+1] for tt_inds in trial_inds_inc_types]
# for i_type, tt_inds in enumerate(trial_inds_inc_types):
#     tt_resps = X_means[i_cell, tt_inds, i_period+1]
#     tt_period_mean.append(tt_resps)

# test whether an individual cell differs across trial types
# H, p = stats.kruskal(*tt_period_mean)
# discrim[i_cell, i_period] = p
# test whether an individual cell correlates with avg reward
# i_tt_period_mean = []
# m, b, r, p, stderr = stats.linregress([np.mean(x) for x in tt_period_mean], mean_rews)
# rew_correls[i_cell, i_period, :] = [m, r, p]


def plot_hist(data, df, ax=None):
    # modified from Dabney et al., 2020
    xlim = (-13, 13)
    height, bin_edges = np.histogram(data[~np.isnan(data)], density=True, bins=xlim[1] - xlim[0], range=xlim)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    if ax is not None:
        ax.bar(centers - width / 2., height, width=width,
               color=plt.cm.plasma(stats.norm.cdf(centers, scale=3)),
               edgecolor='w', linewidth=2)

    xs = np.linspace(-13., 13., 2000)
    ax.plot(xs, stats.t.pdf(xs, df), color='k', linewidth=6, alpha=1.)
    ax.set_xlim(xlim)
    ax.set_xlabel("t-Statistic", fontsize=14)
    ax.set_ylabel("Relative frequency", fontsize=14)
    plt.tight_layout()


def plot_cs_cs_corr(X_tt_means, periods, var_types):
    # ODR and correlation of each cell's activity between different CSs
    """
    :param X_tt_means: array of shape n_trial_types x n_neurons x n_periods
    :param periods: dictionary containing info about periods
    :param var_types: CSs with non-deterministic (variable) outcomes
    :return:
    """
    fig, axs = plt.subplots(periods['n_periods_to_plot'], 3, figsize=(10, periods['n_periods_to_plot'] * 3),
                            squeeze=False)
    for h, h_period in enumerate(periods['periods_to_plot']):
        col = 0
        for i, i_type in enumerate(var_types):
            cell_means_tt_i = X_tt_means[i_type, :, h_period]
            for j in range(i):
                ax = axs[h, col]
                cell_means_tt_j = X_tt_means[var_types[j], :, h_period]
                compute_odr(cell_means_tt_i, cell_means_tt_j, ax)
                # ax.scatter(cell_means_tt_i, cell_means_tt_j)
                # r, p = stats.pearsonr(cell_means_tt_i, cell_means_tt_j)
                #
                # # use orthogonal distance regression to account for error in both x and y
                # linear = odr.Model(linear_func)
                # dat = odr.RealData(cell_means_tt_i, cell_means_tt_j, sx=np.std(cell_means_tt_i),
                #                    sy=np.std(cell_means_tt_j))
                # this_odr = odr.ODR(dat, linear, beta0=[1., 0.])
                # odr_out = this_odr.run()
                # df = odr_out.iwork[10]  # len(cell_means_tt_i) - 2, because 2 parameters
                # # t_stat = odr_out.beta[0]/odr_out.sd_beta[0]  # compute t statistic on the slope parameter
                # # p_val = t.sf(np.abs(t_stat), df)*2
                # plot_points = np.array([np.amin(cell_means_tt_i), np.amax(cell_means_tt_i)])
                # ax.plot(plot_points, linear_func(odr_out.beta, plot_points), 'k--', lw=1)
                # ax.set_title(r'$m={:3.2f}, r={:.3f}$'.format(odr_out.beta[0], r), fontsize=12)
                ax.set_xlabel('CS {} {} Activity'.format(i_type, periods['period_names'][h_period]))
                ax.set_ylabel('CS {} {} Activity'.format(var_types[j], periods['period_names'][h_period]))
                col += 1
    fig.tight_layout()
    hide_spines()


def compute_odr(x, y, ax):
    ax.scatter(x, y)
    r, pearson_p = stats.pearsonr(x, y)

    linear = odr.Model(linear_func)
    dat = odr.RealData(x, y, sx=np.std(x), sy=np.std(y))
    this_odr = odr.ODR(dat, linear, beta0=[0., 0.])
    odr_out = this_odr.run()
    df = odr_out.iwork[10]  # len(cell_means_tt_i) - 2, because 2 parameters
    t_stat = odr_out.beta[0] / odr_out.sd_beta[0]  # compute t statistic on the slope parameter
    odr_p = stats.t.sf(np.abs(t_stat), df) * 2
    plot_points = np.array([np.amin(x), np.amax(x)])
    ax.plot(plot_points, linear_func(odr_out.beta, plot_points), 'k--', lw=1)
    ax.set_title(r'ODR $m={:3.2f}, p={:.3e}$;'.format(odr_out.beta[0], odr_p) + '\n' +
                 r'Pearson $r={:.3f}, p={:.3f}$'.format(r, pearson_p), fontsize=12)


def plot_pairs(rescale, periods, protocol_info, use_taus, save=False):
    """
	:param rescale: n_trial_types x n_periods x n_use_cells array of normalized responses
	:param periods: dictionary
	:param protocol_info: dictionary
	:param use_taus: boolean array (n_use_cells,) mask
	:param save: boolean, whether or not to save
	:return:
	"""
    # compare rescaled value to one another
    norm_fig, norm_axs = plt.subplots(periods['n_periods_to_plot'], 3, figsize=(10, periods['n_periods_to_plot'] * 3),
                                      squeeze=False)
    label = 'norm. resp.'
    for h, h_period in enumerate(periods['periods_to_plot']):
        col = 0
        for i, i_type in enumerate(protocol_info['var_types']):
            for j in range(i):
                ax = norm_axs[h, col]
                compute_odr(rescale[i, h_period, use_taus], rescale[j, h_period, use_taus], ax)
                # ax.scatter(t_stats[i][kw][:, h_period], t_stats[j][kw][:, h_period])
                # grouped = np.concatenate((t_stats[i][kw][:, h_period], t_stats[j][kw][:, h_period]))
                # ax.plot(np.linspace(np.amin(grouped), np.amax(grouped), 11),
                #         np.linspace(np.amin(grouped), np.amax(grouped), 11), 'k--')
                ax.set_xlabel('CS {} {}'.format(protocol_info['trial_type_names'][i_type], label))
                ax.set_ylabel('CS {} {}'.format(protocol_info['trial_type_names'][i_type - i + j], label))
                # if col == 1:
                # 	ax.set_title(periods['period_names'][h_period])
                col += 1
    norm_fig.tight_layout()
    hide_spines()
    if save:
        plt.savefig('norm_fig.svg', bbox_inches='tight')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def plot_t(i, cell_means_norm_i, tt_activity, n_total_cells, periods, protocol_info, t_axs):
    """
	:param i: index of var_types
	:param cell_means_norm_i: n_total_cells x n_periods, containing the mean of each cell in each period
	:param tt_activity: list of length n_total_cells, containing arrays of shape n_trials_of_type_i x n_periods
	:param n_total_cells: number of cells to include
	:param periods: dictionary containing info about periods to use, names, etc.
	:param protocol_info: dictionary containing info about protocol
	:param t_axs: axes on which to plot
	:return:
	"""

    # estimate tau following Dabney et al., 2020, supplement pg. 9
    # sometimes useful for these to be nanmin and nanmax
    these_taus = cell_means_norm_i - np.amin(cell_means_norm_i, axis=0)
    these_taus /= np.amax(cell_means_norm_i, axis=0) - np.amin(cell_means_norm_i, axis=0)

    # "ANOVA results where we evaluate the null hypothesis that all cells' normalized responses have the same mean"
    print(stats.f_oneway(*tt_activity, axis=0))
    print(stats.kruskal(*tt_activity, axis=0))

    #     # Creates array of n_cells x n_periods. THIS WAY IS WRONG!!
    #     norm_resps_avg_all_cells = np.mean(cell_means_norm_i, axis=0)  # average across cells. Creates (n_periods,)
    #     norm_resps_sem_all_cells = stats.sem(cell_means_norm_i, axis=0)  # sem across cells
    #     old_t_stat = (cell_means_norm_i - norm_resps_avg_all_cells) / norm_resps_sem_all_cells
    #     print(old_t_stat)

    # following the approach in CalculateKinkiness.m, from Dabney et al., 2020 to compute t-statistics for each cell
    t_stat = np.zeros((n_total_cells, periods['n_periods_to_plot']))
    p_val = np.zeros((n_total_cells, periods['n_periods_to_plot']))
    dofs = np.zeros((n_total_cells, periods['n_periods_to_plot']))

    for i_cell in range(n_total_cells):
        # use the empirical 50%
        diff_from_interp = tt_activity[i_cell] - np.median(
            np.array([np.mean(tt_activity[j_cell], axis=0) for j_cell in range(n_total_cells)]))
        # use the interpolated 50%
        # diff_from_interp = c(iCell,:) - (c10m(iCell) + c90m(iCell)) / 2;   % use the interpolated 50%
        t_stat[i_cell], p_val[i_cell] = stats.ttest_1samp(diff_from_interp[:, periods['periods_to_plot']], 0)
        dofs[i_cell] = tt_activity[i_cell].shape[0] - 1

    # even if this the t-stats are computed within cells (across trials), the distribution
    # of t-stats is taken across neurons, so this remains correct. To quote Dabney et al., 2020,
    # "These  t-statistics  would  be  t-distributed  if  the  differences  between  cells  were  due  to  chance."
    df = n_total_cells - 1

    # 	p = (1. - stats.t.cdf(np.abs(old_t_stat), df)) * 2  # n_cells x n_periods
    optimistic = np.nonzero(np.logical_and(p_val < periods['alpha'], t_stat > 0))
    pessimistic = np.nonzero(np.logical_and(p_val < periods['alpha'], t_stat < 0))
    #     print(optimistic, pessimistic)

    for j, j_period in enumerate(periods['periods_to_plot']):
        # plot vs. t-distribution with corresponding degrees of freedom
        ax = t_axs[j, i]
        plot_hist(t_stat[:, j], df, ax)
        if i == 1:
            ax.set_title(periods['period_names'][j_period], fontsize=14)

    out = {'taus': these_taus,
           'x_norm': cell_means_norm_i,
           't': t_stat,
           'p': p_val,
           'opt': optimistic,
           'pess': pessimistic}

    return out


def parallel_facemap_decode(ret, dec_per, pop_types=['simul'], classifiers=[SVC(kernel='linear')], n_time_bins=24):

    # the first part is common to recording during both the trace period, in which case we are focusing on
    # the SameRewDist task and contrasting within/across distribution pairings with the same mean; and during the
    # reward period, in which case we are focusing on the Higher Moments task and looking at 2 and 6 uL
    # rewards in the low-variance (CS3) and high-variance (CSs 4/5) cases

    if dec_per == 'trace':  # late trace period
        start_fm = 2
        end_fm = 3
    elif dec_per == 'rew':  # reward period
        start_fm = 3
        end_fm = 4
    else:
        raise Exception('dec_per not recognized')

    time_bins = np.linspace(-1, 5, n_time_bins + 1)
    bin_width = np.mean(np.diff(time_bins))
    n_jobs = 1  # can't use multiprocessing in nested fashion, and since this is parallel, just use 1 job here

    # get mouse/session info
    mouse_name = ret['name']
    file_date = str(ret['exp_date'])  # may fail for very old sessions, which require file_date_id
    protocol = ret['protocol']

    colors, protocol_info, periods, kwargs = load_params(protocol)

    paths = get_db_info()
    fig_path = os.path.join(paths['behavior_fig_roots'][0], mouse_name, file_date)
    key = '_'.join([mouse_name, file_date])
    print(key)

    # load trial types
    raw = loadmat(ret['raw_data_path'])
    trial_types = raw['SessionData']['TrialTypes'] - 1
    #     perm = rng.permutation(trial_types)
    rews = raw['SessionData']['RewardDelivered']
    rews = rews[~np.isnan(rews)]

    # load facemap data
    facemap_path = os.path.join(paths['facemap_root'], mouse_name, file_date)
    facemap_p = os.path.join(facemap_path, '_'.join([mouse_name, file_date, 'facemap.p']))
    facemap = DataStream(facemap_p, 'facemap')

    try:
        # assemble facemap data into array of shape n_predictors (usually 52 or 53) x n_trials x nsamps
        # assumes that mot_svd always exists, which it should!
        facemap.mat_raw = np.stack(
            (*[facemap.dat['dat'][m][:, facemap.start_ind:facemap.end_ind] for m in
               ['whisking', 'running', 'pupil'] if m in facemap.dat['dat'].keys()],
             *np.transpose(facemap.dat['dat']['mot_svd'][:, facemap.start_ind:facemap.end_ind, :],
                           (2, 0, 1))
             ), axis=0)

    except:
        # mot_svd doesn't exist OR neither whisking, running, nor pupil exists. In this case, don't even
        # run the decoding, for consistency
        print("Couldn't find anything?")
        return

    #     except ValueError:  # neither whisking, running, nor pupil exists
    #         facemap.mat_raw = np.transpose(facemap.dat['dat']['mot_svd'][:, facemap.start_ind:facemap.end_ind, :],
    #                                        (2, 0, 1))

    #     except KeyError:  # mot_svd doesn't exist
    #         facemap.mat_raw = np.stack([facemap.dat['dat'][m][:, facemap.start_ind:facemap.end_ind] for m in
    #                                     ['whisking', 'running', 'pupil'] if m in facemap.dat['dat'].keys()])

    # load behavior data
    behavior_path = os.path.join(paths['behavior_fig_roots'][0], mouse_name, file_date)
    behavior_p = os.path.join(behavior_path, '_'.join([mouse_name, protocol, file_date, str(ret['exp_time']) + '.p']))
    behavior = DataStream(behavior_p, 'behavior')
    behavior.mat_raw = behavior.dat['licks_raw'][np.newaxis, :, behavior.start_ind:behavior.end_ind]

    # will end up being n_predictors x n_trials x n_time_bins
    combined_mat = np.empty((0, facemap.mat_raw.shape[1], n_time_bins))
    combined_mean = np.empty((facemap.mat_raw.shape[1], 0))

    # all_trial_types.append(trial_types)  # XXX: do something about this

    for stream, agg in zip([facemap, behavior], [np.nanmean, np.nansum]):
        #     for stream, agg in zip([facemap], [np.nanmean]):
        bin_inds = np.digitize(stream.time[stream.start_ind:stream.end_ind], time_bins) - 1
        # behavior throws Mean of empty slice warning because Unexpected Reward trials have NaNs
        bin_raw = np.stack(
            [agg(stream.mat_raw[..., ind == bin_inds], axis=-1) / bin_width for ind in np.arange(n_time_bins)], axis=-1)
        combined_mat = np.concatenate((combined_mat, bin_raw), axis=0)

        start_idx = np.argmin(np.abs(stream.time[stream.start_ind:stream.end_ind] - start_fm))
        end_idx = np.argmin(np.abs(stream.time[stream.start_ind:stream.end_ind] - end_fm))
        combined_mean = np.concatenate(
            (combined_mean, agg(stream.mat_raw[..., start_idx:end_idx], axis=2).T / (end_fm - start_fm)), axis=1)

    # all_combined_mats.append(combined_mat)  # XXX: do something about this

    po = np.argsort(protocol_info['mean'][:protocol_info['n_trace_types']])
    tt_names = np.array(protocol_info['trace_type_names'], dtype='object')

    sess_result = {}
    sess_df = {}; mean_df = {}

    # This is the part that differs between the two periods
    if dec_per == 'trace':
        n_splits = 5
        cong_keys = ['Congruent', 'Incongruent 1', 'Incongruent 2']
        for method in ['pair', 'cong', 'ccgp']:
            sess_result[method] = {}
            for df in [sess_df, mean_df]:
                df[method] = {'mouse': [], 'clf': [], 'session': [], 'grouping': [], 'Accuracy': [], 'shuff': []}
            print(method)
            if method == 'pair':
                for i, tt_i in enumerate(kwargs['same_avg_rew']):
                    for j, tt_j in enumerate(kwargs['same_avg_rew'][:i]):
                        inds = np.isin(trial_types, [tt_i, tt_j])
                        grouping = tt_names[tt_j] + 'v.' + tt_names[tt_i]
                        # print(grouping, combined_mean[inds].shape, trial_types[inds].shape)
                        sess_df, mean_df, score_dict = store_results(sess_df, mean_df, combined_mean[inds],
                                                                     trial_types[inds], pop_types, classifiers, method,
                                                                     key, grouping, mouse_name, n_jobs, n_splits)
                        sess_result[method][grouping] = score_dict

            elif method == 'cong':
                for grouping in cong_keys:
                    if grouping == 'Congruent':
                        inds_a = np.isin(trial_types, [2, 3])
                        inds_b = np.isin(trial_types, [4, 5])
                    elif grouping == 'Incongruent 1':
                        inds_a = np.isin(trial_types, [2, 4])
                        inds_b = np.isin(trial_types, [3, 5])
                    elif grouping == 'Incongruent 2':
                        inds_a = np.isin(trial_types, [2, 5])
                        inds_b = np.isin(trial_types, [3, 4])
                    X = np.concatenate((combined_mean[inds_a], combined_mean[inds_b]), axis=0)
                    y = np.concatenate((np.zeros(np.sum(inds_a)), np.ones(np.sum(inds_b))))
                    # print(grouping, X.shape, y.shape)
                    sess_df, mean_df, score_dict = store_results(sess_df, mean_df, X, y, pop_types, classifiers, method,
                                                                 key, grouping, mouse_name, n_jobs, n_splits)
                    sess_result[method][grouping] = score_dict

            # elif method == 'ccgp':

    elif dec_per == 'rew':
        baseline_tt = 1
        probe_rews = [2, 6]
        n_splits = 3  # sometimes very few example trials
        for rew in probe_rews:
            sess_result[rew] = {}
            for df in [sess_df, mean_df]:
                df[rew] = {'mouse': [], 'clf': [], 'session': [], 'grouping': [], 'Accuracy': [], 'shuff': []}
            print(rew)
            for tt in [2, 3]:
                combo_tt_1 = np.logical_and(rews == rew, trial_types == baseline_tt)
                combo_tt_2 = np.logical_and(rews == rew, trial_types == tt)
                X = np.concatenate((combined_mean[combo_tt_1], combined_mean[combo_tt_2]), axis=0)
                y = np.concatenate((np.zeros(np.sum(combo_tt_1)), np.ones(np.sum(combo_tt_2))))
                grouping = (tt_names[po == baseline_tt] + 'v.' + tt_names[po == tt]).item()
                # print(grouping, X.shape, y.shape)
                sess_df, mean_df, score_dict = store_results(sess_df, mean_df, X, y, pop_types, classifiers,
                                                             rew, key, grouping, mouse_name, n_jobs, n_splits)
                sess_result[rew][grouping] = score_dict

    return sess_result, sess_df, mean_df, trial_types, combined_mat


def store_results(all_df, mean_df, X, y, pop_types, classifiers, method, key, grouping, mouse_name, n_jobs, n_splits):
    score_dict = flexible_decode(X, y, pop_types, classifiers, n_jobs, n_splits)
    for i_clf, clf in enumerate(classifiers):
        n_scores = len(score_dict['simul']['results'][i_clf])
        # print(n_scores, score_dict['simul']['results'][i_clf])
        all_df[method]['mouse'].extend([mouse_name] * n_scores)
        all_df[method]['clf'].extend([i_clf] * n_scores)
        all_df[method]['session'].extend([key] * n_scores)
        all_df[method]['grouping'].extend([grouping] * n_scores)
        all_df[method]['Accuracy'].extend(score_dict['simul']['results'][i_clf])
        all_df[method]['shuff'].extend(score_dict['simul']['shuff'][i_clf])

        mean_df[method]['mouse'].append(mouse_name)
        mean_df[method]['clf'].append(i_clf)
        mean_df[method]['session'].append(key)
        mean_df[method]['grouping'].append(grouping)
        mean_df[method]['Accuracy'].append(np.mean(score_dict['simul']['results'][i_clf]))
        mean_df[method]['shuff'].append(np.mean(score_dict['simul']['shuff'][i_clf]))

    return all_df, mean_df, score_dict


def flexible_decode(X, y, pop_types=['simul'], clfs=[SVC(kernel='linear')], n_jobs=-1, n_splits=5):
    """
    :param X: data matrix. n_trials x n_neurons or n_trials x n_facemap_predictors
    :param y: target outputs. shape n_trials.
    :param pop_types: list of 'simul' (for actual trials) and/or 'pseudopop' (breaking trial correlations)
    :param clfs: list of classifiers to use (in the event of comparison)
    :return:
    """
    # iterate over classifiers, performing K-fold cross-validation
    scoring = 'balanced_accuracy'
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    n_trial_types = len(np.unique(y))
    rng = np.random.default_rng(seed=1)

    all_results = {}

    for pop_type in pop_types:

        results = []
        shuff_results = []
        ps = []

        this_pop = X.copy()
        if pop_type == 'pseudopop':  # create pseudopopulation by randomly sampling trials of a given type
            for i_type in range(n_trial_types):
                # this_pop[y == i_type] = rng.permutation(this_pop[y == i_type])  # I confirmed this works
                this_pop[y == i_type] = rng.choice(this_pop[y == i_type], replace=True)

        perm = rng.permutation(y)
        for clf in clfs:  # loop through classifiers

            scores = cross_val_score(clf, this_pop, y, cv=kfold, scoring=scoring, n_jobs=n_jobs)
            shuff = cross_val_score(clf, this_pop, perm, cv=kfold, scoring=scoring, n_jobs=n_jobs)

            results.append(scores)
            shuff_results.append(shuff)

            tstat, tp = stats.ttest_rel(scores, shuff)
            ps.append(tp)

        all_results[pop_type] = {'results': results, 'shuff': shuff_results, 'ps': ps}

    return all_results


def sklearn_decode(use_mean, trials, shuff_trials, label, pop_types, trial_types, perm, all_clf, classifiers, names,
                   kfold, key, confusion_fig_path, comparison_fig_path, scoring='balanced_accuracy'):
    """
    :param use_mean: n_trials x n_neurons or n_trials x n_facemap_predictors
    :param trials: boolean array saying whether or not to use the trial (excluded if e.g. Unexpected Reward)
    :param shuff_trials: boolean array saying whether or not to use the permuted trial
    :param label: e.g. 'include' or 'same', used for saving
    :param pop_types: list: ['simul', 'pseudopop'] for simultaneous or pseudopopulation decoding
    :param trial_types: integer array of shape (n_trials,) containing the trial type of each trial
    :param perm: permuted version of trial_types
    :param all_clf: dictionary in which to stash results from multiple loops
    :param classifiers: the sklearn classifiers to use
    :param names: names of the sklearn classifiers
    :return:
    """

    n_trial_types = len(np.unique(trial_types))
    rng = np.random.default_rng()

    n_clf = len(classifiers)
    ncols = 4
    nrows = int(np.ceil(n_clf / ncols))

    if ~np.any(trials):  # or len(trials) != neural_mean.shape[1]:  # why should the latter part be necessary?
        return

    for pop_type in pop_types:

        trace_results = []
        shuff_results = []
        ps = []
        confusions = []

        inc_trials = trial_types[trials]
        n_active_types = len(np.unique(inc_trials))

        if pop_type == 'simul':  # normal population decoding
            this_pop = use_mean[trials]
        elif pop_type == 'pseudopop':  # create pseudopopulation by randomly sampling trials of a given type
            this_pop = use_mean[trials].copy()
            for i_type in range(n_trial_types):
                this_pop[inc_trials == i_type] = rng.permutation(this_pop[inc_trials == i_type])  # I confirmed this works

        fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))
        for i in range(ncols * nrows):  # loop through classifiers

            ax = axs.flat[i]
            if i < len(names):

                name = names[i]
                clf = classifiers[i]

                scores = cross_val_score(clf, this_pop, inc_trials, cv=kfold, scoring=scoring, n_jobs=-1)
                shuff = cross_val_score(clf, this_pop, perm[shuff_trials], cv=kfold, scoring=scoring, n_jobs=-1)

                trace_results.append(scores)
                shuff_results.append(shuff)

                tstat, tp = stats.ttest_rel(scores, shuff)
                ps.append(tp)

                # compute cv predictions, and thus the confusion matrix
                y_pred = cross_val_predict(clf, this_pop, inc_trials, cv=kfold, n_jobs=-1)
                confusion = confusion_matrix(inc_trials, y_pred)
                confusions.append(confusion)
                im = plot_confusion(confusion, ax, name, True)
                if i % ncols == 0:
                    ax.set_ylabel('True Label')
            else:
                ax.axis('off')

        add_cbar(fig, im, '')
        hide_spines()
        conf_fparts = os.path.basename(confusion_fig_path).split('.')
        conf_name = '_'.join([conf_fparts[0], pop_type]) + '.' + conf_fparts[1]
        fig.savefig(os.path.join(os.path.dirname(confusion_fig_path), conf_name), bbox_inches='tight', dpi=300)

        all_clf[label][pop_type][key] = {'trace': trace_results,
                                         'shuff': shuff_results,
                                         'ps': ps,
                                         'confusion': confusions}

        # boxplot algorithm comparison
        plot_box(np.array(trace_results).T, np.array(shuff_results).T, names, ps)
        comp_fparts = os.path.basename(comparison_fig_path).split('.')
        comp_name = '_'.join([comp_fparts[0], pop_type]) + '.' + comp_fparts[1]
        plt.savefig(os.path.join(os.path.dirname(comparison_fig_path), comp_name), bbox_inches='tight', dpi=300)

    return all_clf

def plot_decode_means(all_clf, label, pop_types, names, fpath):

    n_clf = len(names)
    if n_clf == 1:
        names = ['']  # avoid plotting title
        ncols = 1  # as long as I'm only doing one classifier
    else:
        ncols = 4
    nrows = int(np.ceil(n_clf / ncols))

    # plot confusion matrices
    for pop_type in pop_types:
        # n_rets x n_clf x n_trial_types x n_trial_types
        if all_clf[label][pop_type]:
            all_confusion = np.array(
                [all_clf[label][pop_type][k]['confusion'] for k in all_clf[label][pop_type].keys()])
            norm_confusion = all_confusion / np.sum(all_confusion, axis=-1, keepdims=True)
            mean_confusion = np.mean(norm_confusion, axis=0)  # average across rets

            fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 3), squeeze=False)
            for i in range(ncols * nrows):
                ax = axs.flat[i]
                if i < len(names):
                    im = plot_confusion(mean_confusion[i, :, :], ax, names[i])
                if i % ncols == 0:
                    ax.set_ylabel('True Label')
                else:
                    ax.axis('off')
            add_cbar(fig, im, '')

            # boxplot
            ordered_means = [all_clf[label][pop_type][k]['trace'] for k in all_clf[label][pop_type].keys()]
            shuff_means = [all_clf[label][pop_type][k]['shuff'] for k in all_clf[label][pop_type].keys()]
            grand_means = np.array([[np.mean(x) for x in arr] for arr in ordered_means])
            grand_shuff = np.array([[np.mean(x) for x in arr] for arr in shuff_means])
            grand_t, grand_p = stats.ttest_rel(grand_means, grand_shuff, axis=0)

            # boxplot algorithm comparison
            plot_box(grand_means, grand_shuff, names, grand_p)
            fparts = os.path.basename(fpath).split('.')
            fname = '_'.join([fparts[0], pop_type]) + '.' + fparts[1]
            plt.savefig(os.path.join(os.path.dirname(fpath), fname), bbox_inches='tight', dpi=300)

def temporal_lr(bin_centers, all_mats, all_trial_types, colors, exclude_tt, fpath, reg_C=1.0):
    """
    :param bin_centers: 1-D array containing centers of time bins
    :param all_mats: list of length n_sessions (or however many decoding runs to perform). Each element is a 3-D array
    of shape n_neurons x n_trials x n_time_bins
    :param all_trial_types: list of length n_sessions (or however many decoding runs to perform). Each element is a 1-D
    array of shape n_trials
    :param colors: dictionary of colors
    :param exclude_tt: trial types to exclude
    :param fpath: path to save
    :return:
    """
    # Investigate decoding a bit more carefully, across timepoints
    # iterate over time bins, performing K-fold cross-validation in each bin
    n_splits = 4
    scoring = 'balanced_accuracy'
    name = 'Support Vector Machine'
    clf = SVC(C=reg_C)
    n_bins = len(bin_centers)

    all_tmean = np.zeros((n_bins, len(all_mats)))
    plt.figure(figsize=(3, 2))

    for j, (trial_types, tm) in enumerate(zip(all_trial_types, all_mats)):

        t_decoding = np.zeros((n_bins, n_splits))
        inc_trials = ~np.isin(trial_types, exclude_tt)  # mask to exclude unpredicted reward
        label = 'include'

        if ~np.any(inc_trials):
            break

        for i in range(n_bins):
            start = time.time()
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
            t_decoding[i, :] = cross_val_score(clf, tm[:, inc_trials, i].T, trial_types[inc_trials], cv=kfold,
                                               scoring=scoring, n_jobs=-1)
            if i % 50 == 0:
                print("%s: %f (%f) in %fs" % (
                name, t_decoding[i, :].mean(), t_decoding[i, :].std(), time.time() - start))

        # ftime = neural.time[neural.start_ind:neural.end_ind]
        t_mean = np.mean(t_decoding, axis=1)
        t_sem = stats.sem(t_decoding, axis=1)
        all_tmean[:, j] = t_mean

        plt.plot(bin_centers, t_mean, 'b-', lw=1, alpha=.3)
    #         plt.fill_between(tile_time, t_mean + t_sem, t_mean - t_sem, color='b', alpha=0.1)

    all_mean = np.mean(all_tmean, axis=1)
    all_sem = stats.sem(all_tmean, axis=1)
    plt.plot(bin_centers, all_mean, 'b-', lw=3)
    plt.fill_between(bin_centers, all_mean + all_sem, all_mean - all_sem, color='b', alpha=.4)
    n_active_types = len(np.unique(trial_types[inc_trials]))
    plt.axhline(1. / n_active_types, ls='--', c=colors['vline_color'])
    plt.axvspan(0, 1, color=colors['vline_color'], alpha=0.2)
    plt.axvline(x=3, c=colors['vline_color'], alpha=0.2, ls='--')
    plt.xlabel('Time from CS')
    plt.ylabel('Decoding Accuracy')
    hide_spines()
    nameparts = name.split()
    fparts = os.path.basename(fpath).split('.')
    fname = os.path.join(os.path.dirname(fpath), '_'.join([fparts[0], label, *nameparts]) + '.' + fparts[1])
    plt.savefig(fname, bbox_inches='tight')
    plt.savefig(os.path.join(os.path.dirname(fpath), '_'.join([fparts[0], label, *nameparts]) + '.png'), bbox_inches='tight', dpi=500)
    return all_tmean


def plot_RDA(X_mat, n_trial_types, period_names, period_inds=[1, 2, 3], metric='euclidean'):
    # Representational Dissimilarity Analysis (RDA) over time
    RDAs = np.zeros((len(period_inds), n_trial_types, n_trial_types))
    #     RDAs = [[]] * len(period_inds)
    for i_per, period_ind in enumerate(period_inds):
        avg_responses = X_mat[:n_trial_types, :, period_ind]
        RDAs[i_per] = pairwise_distances(avg_responses[..., 0], avg_responses[..., 1], metric=metric)
    #         RDAs[i_per] = pairwise_distances(avg_responses, metric=metric)
    #         RDAs[i_per] = np.corrcoef(avg_responses, rowvar=True)
    #         np.fill_diagonal(RDAs[i_per], 0)

    prc_range = np.nanpercentile(RDAs, [2.5, 97.5], axis=None)
    fig, axs = plt.subplots(1, len(period_inds), figsize=(len(period_inds) * 4, 3))
    for i_per, period_ind in enumerate(period_inds):
        ax = axs[i_per]
        im = ax.imshow(RDAs[i_per], cmap='magma', vmin=prc_range[0], vmax=prc_range[1])
        ax.set_title(period_names[period_ind])
    add_cbar(fig, im, '{} distance'.format(metric.capitalize()))
    return RDAs


def print_neuron_info(neuron_info, subset):
    for idx in subset:
        print(neuron_info['names'][idx], neuron_info['file_dates'][idx], neuron_info['neuron_idx_inc'][idx])


def save_neuron_pdfs(neuron_info, subset, pdf_path, label, table='imaging'):
    if not os.path.exists(os.path.dirname(pdf_path)):
        os.makedirs(pdf_path)
    neuron_pdf = fitz.open()
    for idx in subset:
        if table == 'imaging':
            fig_path = os.path.join(neuron_info['fig_paths'][idx], '_'.join([str(neuron_info['names'][idx]),
                                                                             str(neuron_info['file_dates'][idx]),
                                                                             'chan',
                                                                             'Ca', label, 'neurons.pdf']))
        elif table == 'ephys':
            fig_path = os.path.join(neuron_info['fig_paths'][idx], '_'.join([str(neuron_info['names'][idx]),
                                                                             str(neuron_info['file_dates'][idx]),
                                                                             'spikes.pdf']))
        session_pdf = fitz.open(fig_path)
        neuron_pdf.insertPDF(session_pdf, from_page=int(neuron_info['neuron_idx_inc'][idx]),
                             to_page=int(neuron_info['neuron_idx_inc'][idx]))
        session_pdf.close()
    neuron_pdf.save(pdf_path)
    neuron_pdf.close()


def consolidate_decode(data, fig=1, pop_sizes=None, color=None, n_runs=30, ci=True, kern='linear', plot_mat=False,
                       replace=True, reg_C=1.0):
    """
    :param data: cues_resps or residual (n_trial_types x n_cells x max_n_trials array). Here, "n_cells" can also be
    "n_behavior_features"
    :param fig: matplotlib figure to plot to
    :param pop_sizes: 1-D array of pseudopopulation sizes to use, or int, indicating number of sizes to use
    :param color: color to plot in
    :param n_runs: how many runs (with random subsets of data) to perform
    :param ci: whether to plot 95% CI as errorbars. If False, plot s.e.m. instead
    :param kern: kernel to use for SVC. {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’linear’.
    See sklearn docs for details.
    :param plot_mat: whether or not to plot the confusion matrix
    :return:
    # since decoding performance isn't limited by noise correlations, combine cells from different recording sessions
    # into pseudopopulation and see how performance scales with number of neurons
    """
    clf = SVC(C=reg_C, kernel=kern)
    n_pseudo_trials_per_type = 100

    n_trial_types = data.shape[0]
    n_cells = data.shape[1]

    if len(data.shape) == 4:
        n_partitions = 1  # CCGP
    elif len(data.shape) == 3:
        n_partitions = 2  # pull training and test sets from the same conditions
    else:
        raise ValueError('Data must be either 3 or 4-dimensional')

    pop_sizes, n_sizes = get_pop_sizes(pop_sizes, n_cells)
    pop_scores = np.full((n_sizes, n_runs), np.nan)

    # this will end up being the vector of correct classifications
    pseudo_types = np.repeat(np.arange(n_trial_types), n_pseudo_trials_per_type)

    rng = np.random.default_rng(seed=1)
    for i_pop, pop_size in enumerate(pop_sizes):

        for i_run in range(n_runs):

#             if pop_size < n_cells or i_run == 0:  # only run once for pop_size == total_cells to save time

            # pseudo_mat will (eventually) be an array of pseudotrials
            pseudo_mat = np.zeros((n_pseudo_trials_per_type * n_trial_types, pop_size, 2))

            # each run of the decoder gets its own consistent pseudopopulation
            # sample with replacement, i.e. bootstrapping
            rew_cells_to_use = np.random.choice(n_cells, size=pop_size, replace=replace)

            for it, cell in enumerate(rew_cells_to_use):
                trial_resps = data[:, cell]  # get all trials for this cell
                trials_per_type = ~np.isnan(trial_resps)  # boolean array

                # indices of trials to use for each cell in this pseudotrial, ignoring nans
                for i_partition in range(2):
                    #                         print(np.sum(trials_per_type[..., i_partition], axis=1))
                    if n_partitions == 1:
                        trials = [rng.choice(np.arange(tt), size=n_pseudo_trials_per_type) for tt in
                                  np.sum(trials_per_type[..., i_partition], axis=1)]
                        pseudo_mat[:, it, i_partition] = np.concatenate(
                            [trial_resps[i_type, trials_per_type[i_type, :, i_partition], i_partition][tt] for
                             i_type, tt in enumerate(trials)])
                    else:
                        trials = [
                            rng.choice(np.arange(i_partition, tt, n_partitions), size=n_pseudo_trials_per_type)
                            for tt in np.sum(trials_per_type, axis=1)]
                        pseudo_mat[:, it, i_partition] = np.concatenate(
                            [trial_resps[i_type, trials_per_type[i_type]][tt] for i_type, tt in enumerate(trials)])

            clf = clf.fit(pseudo_mat[..., 0], pseudo_types)
            pop_scores[i_pop, i_run] = clf.score(pseudo_mat[..., 1], pseudo_types)

            if i_pop == n_sizes - 1 and i_run == 0:
                # compute cv predictions, and thus the confusion matrix
                y_pred = clf.predict(pseudo_mat[..., 1])
                confusion = confusion_matrix(pseudo_types, y_pred)

    # plot decoder accuracy as a function of included cells
    fig1 = plt.figure(fig)
    if ci:
        err = stats.sem(pop_scores, axis=1, nan_policy='omit') * 1.96
    else:
        err = stats.sem(pop_scores, axis=1, nan_policy='omit')
    plt.errorbar(pop_sizes, np.nanmean(pop_scores, axis=1), err,
                 fmt='.-', ms=10, elinewidth=2, lw=2, capsize=5, color=color, ecolor=color)
    # fmt='.-', ms=10, capsize=5, color=color, ecolor=color)
    plt.xlabel('Pseudo-population size')
    plt.ylabel('Decoder accuracy')
    hide_spines()

    if plot_mat:
        rng = np.random.default_rng()
        fig2, ax = plt.subplots(num=rng.integers(low=1000, high=100000))
        im = plot_confusion(confusion, ax, 'Pseudopopulation', True)
        plt.ylabel('True Label')
        plt.colorbar(im)

    return pop_scores, pop_sizes, fig1


def get_pop_sizes(pop_sizes, n_cells):
    if pop_sizes is None:
        n_sizes = 20
        pop_sizes = np.logspace(1, np.log10(n_cells), n_sizes).astype(np.int32)
    elif isinstance(pop_sizes, int):
        n_sizes = pop_sizes
        pop_sizes = np.logspace(1, np.log10(n_cells), n_sizes).astype(np.int32)
    else:
        n_sizes = len(pop_sizes)
    return pop_sizes, n_sizes


def prepare_pseudomat(is_ccgp, contains_test, data, train_per, test_per, sid_inds, n_trial_types, use_splits,
                      n_pseudo_trials_per_type, pseudo_mat, cell_count, rng=np.random.default_rng(seed=1), train_ratio=.75):

    if is_ccgp:
        if contains_test:
            use_data = data[..., 1, test_per]
        else:
            use_data = data[..., 0, train_per]
        subt = validate_ccgp(use_data, sid_inds, n_trial_types)

    else:
        use_data = data[..., train_per]
        subt_all = ~np.isnan(use_data)[:, sid_inds]
        subt = [np.unique(np.nonzero(subt_all[i_type])[1]) for i_type in range(n_trial_types)]
        assert [np.all(np.sum(subt_all, axis=2)[i_type] == len(subt[i_type]) for i_type in range(n_trial_types))]

    for i_type in range(n_trial_types):
        # get a random subset of trials on each session
        # type_folds = np.array_split(rng.permutation(np.arange(subt[i_type, 0])), train_splits)
        if use_splits > 2:
            type_folds = np.array_split(rng.permutation(subt[i_type]), use_splits)
        elif use_splits == 2:
            tmp = rng.permutation(subt[i_type])
            type_folds = [tmp[:int(train_ratio * len(tmp))], tmp[int(train_ratio * len(tmp)):]]
        else:
            type_folds = [rng.permutation(subt[i_type])]
        for i_fold, fold in enumerate(type_folds):
            if contains_test and np.all(
                    fold == type_folds[-1]):  # in case of test set for ccgp or cross-temporal decoding
                i_fold = -1
            if contains_test and np.all(fold == type_folds[-1]) and not is_ccgp:  # cross-temporal decoding
                perm_data = data[..., test_per][i_type, sid_inds][..., rng.choice(fold, n_pseudo_trials_per_type)]
            else:
                # for each subset (fold), put them in a random order the correct number of times
                perm_data = use_data[i_type, sid_inds][..., rng.choice(fold, n_pseudo_trials_per_type)]
            pseudo_mat[i_type * n_pseudo_trials_per_type:(i_type + 1) * n_pseudo_trials_per_type,
                cell_count:cell_count + len(sid_inds), i_fold] = perm_data.T

    return pseudo_mat


def validate_ccgp(use_data, sid_inds, n_trial_types):
    subt_all = np.sum(~np.isnan(use_data), axis=2)[:, sid_inds]
    subt = [np.arange(subt_all[i_type, 0]) for i_type in range(n_trial_types)]
    # confirm that sids are correct and all cells in that session have same number of trials
    assert np.all([subt_all[:, 0] == x for x in subt_all.T])
    return subt


def disjoint_regress(data, sids, means, cell_inds=None, train_per=3, test_per=3, n_splits=5, do_zscore=True, do_cv=True):

    rng = np.random.default_rng(seed=1)
    n_pseudo_trials_per_type = 200
    n_trial_types = data.shape[0]
    n_cells = data.shape[1]
    if cell_inds is None:
        cell_inds = np.arange(n_cells)

    contains_test = False if train_per == test_per else True

    # regress this particular population of cells onto value
    pseudo_mat = np.zeros((n_pseudo_trials_per_type * n_trial_types, len(cell_inds), 1))

    # this will end up being the vector of correct mean values
    pseudo_types = np.repeat(means[:n_trial_types], n_pseudo_trials_per_type)

    cell_count = 0
    for sid in np.unique(sids):
        sid_inds = np.intersect1d(np.flatnonzero(sids == sid), cell_inds)
        if len(sid_inds) > 0:
            pseudo_mat = prepare_pseudomat(False, contains_test, data, train_per, test_per, sid_inds,
                                           n_trial_types, 1, n_pseudo_trials_per_type, pseudo_mat, cell_count, rng=rng)
        cell_count += len(sid_inds)

    # clf = RidgeCV().fit(pseudo_mat[..., 0], pseudo_types)
    # score = clf.score(pseudo_mat[..., 0], pseudo_types)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    scaler = StandardScaler()
    ridge = Ridge() if do_cv else RidgeCV(cv=kfold)  # paradoxical, but RidgeCV will end up running on the entire dataset just once, unlike cross_validate(Ridge())
    if do_zscore:
        pipeline = Pipeline([('transformer', scaler), ('estimator', ridge)])
    else:
        pipeline = Pipeline([('estimator', ridge)])

    if do_cv:
        score_dict = cross_validate(pipeline, pseudo_mat[..., 0], pseudo_types, cv=kfold, return_estimator=True)
        return score_dict['estimator'], np.mean(score_dict['test_score'])
    else:
        return pipeline.fit(pseudo_mat[..., 0], pseudo_types)['estimator'].coef_


def disjoint_decode(data, sids, n_splits=6, pop_sizes=None, kern='linear', train_per=3, test_per=3, reg_C=1.0, do_zscore=True):
    """
    Split dataset into disjoint training sets (plural) and test set (singular). E.g. if n_splits=10, then there will
    be 9 training sets and 1 test set. This allows us to perform statistics as normal, assuming independence.
    :param data: cues_resps or residual (n_trial_types x n_cells x max_n_trials array x n_periods). Here, "n_cells"
    can also be "n_behavior_features". If ccgp, there is an additional dimension before n_periods.
    :param sids: vector of length n_cells saying which session each neuron belongs to so that they can be grouped together
    :param pop_sizes: 1-D array of pseudopopulation sizes to use, or int, indicating number of sizes to use
    :param n_splits: how many disjoint subsets to create
    :param kern: kernel to use for SVC. {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’linear’.
    :param train_per: period to use for training data
    :param test_per: period to use for testing data, in event of cross-temporal decoding
    See sklearn docs for details.
    :return:
    # since decoding performance isn't limited by noise correlations, combine cells from different recording sessions
    # into pseudopopulation and see how performance scales with number of neurons
    """
    n_trial_types = data.shape[0]
    n_cells = data.shape[1]

    scaler = StandardScaler()
    clf = SVC(C=reg_C, kernel=kern) if n_trial_types == 2 else LogisticRegression(C=reg_C, max_iter=2000, multi_class='multinomial', class_weight='balanced')

    if do_zscore:
        pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])
    else:
        pipeline = Pipeline([('estimator', clf)])

    n_pseudo_trials_per_type = 200

    pop_sizes, n_sizes = get_pop_sizes(pop_sizes, n_cells)
    pop_scores = np.full((n_sizes, n_splits - 1), np.nan)
    coefs = np.full((n_sizes, n_splits - 1, int(n_trial_types * (n_trial_types - 1) / 2), max(pop_sizes)), np.nan)
    confusions = np.full((n_sizes, n_splits - 1, n_trial_types, n_trial_types), np.nan)
    cell_inds_all = np.zeros(n_sizes, dtype=object)

    contains_test = False
    if len(data.shape) == 5:  # CCGP
        is_ccgp = True
        # with CCGP, we don't use the same conditions (therefore trials) for training and testing. Therefore, we can
        # use all the train conditinos as training data and all the test conditions as testing data. n_splits -1 is correct.
        train_splits = n_splits - 1
    elif len(data.shape) == 4:  # pull training and test sets from the same conditions
        train_splits = n_splits
        is_ccgp = False
        if train_per != test_per:
            contains_test = True
    else:
        raise ValueError('Data must be either 5 or 4-dimensional')

    # this will end up being the vector of correct classifications
    pseudo_types = np.repeat(np.arange(n_trial_types), n_pseudo_trials_per_type)

    for i_pop, pop_size in enumerate(pop_sizes):

        rng = np.random.default_rng(seed=1)
        cell_inds = rng.choice(n_cells, size=pop_size, replace=False)
        # print(cell_inds)
        cell_inds_all[i_pop] = []

        # pseudo_mat will (eventually) be an array of pseudotrials
        pseudo_mat = np.zeros((n_pseudo_trials_per_type * n_trial_types, pop_size, n_splits))

        cell_count = 0
        for sid in np.unique(sids):
            sid_inds = np.intersect1d(np.flatnonzero(sids == sid), cell_inds)
            cell_inds_all[i_pop].extend(sid_inds)
            if len(sid_inds) > 0:

                pseudo_mat = prepare_pseudomat(is_ccgp, contains_test, data, train_per, test_per, sid_inds, n_trial_types,
                                               train_splits, n_pseudo_trials_per_type, pseudo_mat, cell_count, rng=rng)

                if is_ccgp:

                    pseudo_mat = prepare_pseudomat(is_ccgp, True, data, train_per, test_per, sid_inds, n_trial_types, 1,
                                                   n_pseudo_trials_per_type, pseudo_mat, cell_count, rng=rng)

            cell_count += len(sid_inds)

        for i_split in range(n_splits - 1):
            pipeline.fit(pseudo_mat[..., i_split], pseudo_types)
            pop_scores[i_pop, i_split] = pipeline.score(pseudo_mat[..., n_splits - 1], pseudo_types)
            if kern == 'linear' and n_trial_types == 2:
                coefs[i_pop, i_split, :, :pop_size] = pipeline.named_steps['estimator'].coef_
            if n_trial_types > 2:
                preds = pipeline.predict(pseudo_mat[..., n_splits - 1])
                confusions[i_pop, i_split, :, :] = confusion_matrix(pseudo_types, preds)

    return pop_scores, coefs, cell_inds_all, confusions


def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)


def simultaneous_decode(data, train_per, test_per, kern='linear', reg_C=1.0, do_zscore=True, plot=False, label=None, shuffle=False):
    """
    :param data: cues_resps or residual (n_trial_types x n_cells x max_n_trials array x n_periods). Here, "n_cells" can also be
    "n_behavior_features". Since it is simultaneous, it assumes that the number of not nan elements is the same for
    each cell, though not necessarily for each trial type
    :train_per: period to use for training
    :test_per: periods to use for testing
    :param kern: kernel to use for SVC. {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’linear’.
    See sklearn docs for details.
    :return: mean score across CV folds
    """
    scoring = 'balanced_accuracy'
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits)
    rng = np.random.default_rng(seed=1)
    n_jobs = -1  # start in parallel unless memory errors

    n_trial_types = data.shape[0]
    n_cells = data.shape[1]
    max_n_trials = data.shape[2]
    n_sample_trials = 200

    train_data = data[..., train_per]
    test_data = data[..., test_per]

    scaler = StandardScaler()
    if n_trial_types == 2:
        clf = SVC(C=reg_C, kernel=kern, class_weight='balanced')
    else:
        clf = LogisticRegression(C=reg_C, max_iter=2000, multi_class='multinomial', class_weight='balanced')

    if do_zscore:
        pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])
    else:
        pipeline = Pipeline([('estimator', clf)])

    coefs = np.full((n_splits, int(n_trial_types * (n_trial_types - 1) / 2), n_cells), np.nan)
    confusion = None
    correct = None
    # confusions = np.full((n_splits, n_trial_types, n_trial_types), np.nan)

    if len(train_data.shape) == 4:  # CCGP

        y = np.repeat(np.arange(n_trial_types), n_sample_trials)

        notnan_inds = ~np.isnan(train_data)
        assert (np.all(notnan_inds[:, 0, np.newaxis, :] == notnan_inds))
        notnan_inds = notnan_inds[:, 0, :, :]  # shape (2, n_max_trials_per_type, 2), where dim 0 = class A or B and dim 2 = train vs. test

        X_full = np.zeros((n_trial_types, n_trial_types * n_sample_trials, n_cells))
        for i_tt in range(n_trial_types):
            for j_tt, which_data in enumerate([train_data, test_data]):
                which_trial = which_data[i_tt, :, rng.choice(np.flatnonzero(notnan_inds[i_tt, :, j_tt]),
                                                             size=n_sample_trials, replace=True), j_tt]
                X_full[j_tt, i_tt * n_sample_trials:(i_tt + 1) * n_sample_trials, :] = which_trial

        # One simple random model we consider is a shuffle of the data, in which we assign a new, random
        # condition label to each trial for each neuron independently (in a manner that preserves the total
        # number of trials for each condition). In other words, we randomly permute the condition labels (with
        # values from 1 to 8) across all trials, and repeat this procedure separately for every neuron.
        # Since it's independent for every neuron, it doesn't suffice to shuffle y; I have to scramble X
        # I must scramble along the 0 dimension so I'm actually scrambling together different trial types and not
        # just trials of the same type
        if shuffle:
            X_full[0, :] = scramble(X_full[0, :], axis=0)  # (n_sample_trials * n_trial_types) x n_cells
            X_full[1, :] = scramble(X_full[1, :], axis=0)

        pipeline.fit(X_full[0, :], y)
        score = pipeline.score(X_full[1, :], y)

        if kern == 'linear':
            coefs = pipeline.named_steps['estimator'].coef_

    elif len(train_data.shape) == 3:

        y_dummy = np.repeat(np.arange(n_trial_types), max_n_trials)
        X_train_full = train_data.transpose((1, 0, 2)).reshape(n_cells, n_trial_types * max_n_trials)
        nan_inds = np.isnan(X_train_full)
        assert (nan_inds == nan_inds[0]).all()
        if train_per == test_per:

            X_train = X_train_full[:, ~nan_inds[0]].T
            # print(X_train[:5, :10])
            if shuffle:
                # print(X_train.shape)   # n_trials x n_cells
                X_train = scramble(X_train, axis=0)  # shuffle trials
                # print(X_train[:5, :10])
            # print(X_train[:5, :10])
            cv_results = cross_validate(pipeline, X_train, y_dummy[~nan_inds[0]],
                                        scoring=scoring, cv=cv, return_estimator=True)

            score = np.mean(cv_results['test_score'])

            try:
                y_pred = cross_val_predict(pipeline, X_train_full[:, ~nan_inds[0]].T, y_dummy[~nan_inds[0]], cv=cv, n_jobs=n_jobs)
            except OSError:  # [Errno 12] Cannot allocate memory
                y_pred = cross_val_predict(pipeline, X_train_full[:, ~nan_inds[0]].T, y_dummy[~nan_inds[0]], cv=cv, n_jobs=1)
                n_jobs = 1

            correct = y_pred == y_dummy[~nan_inds[0]]

            if n_trial_types == 2:
                if kern == 'linear':
    #                 print(cv_results)
                    for i_fold, model in enumerate(cv_results['estimator']):
    #                     print(model.__dict__.keys())
    #                     print(model)
                        coefs[i_fold] = model.named_steps['estimator'].coef_

            else:
                # y_pred = cross_val_predict(pipeline, X_train_full[:, ~nan_inds[0]].T, y_dummy[~nan_inds[0]], cv=cv, n_jobs=-1)
                confusion = confusion_matrix(y_dummy[~nan_inds[0]], y_pred)

        else:
            # must manually perform cross-validation because I don't want to reuse the same trial during training and
            # test, even if period is different
            assert (nan_inds == nan_inds[0]).all()
            which_trials = np.flatnonzero(~nan_inds[0])
            folds = np.array_split(rng.permutation(which_trials), n_splits)
            scores = np.zeros(n_splits)
            for i_fold, fold in enumerate(folds):
                # avoid DeprecationWarning by first testing for equal length
                train_trials = np.concatenate([f for f in folds if len(f) != len(fold) or ~np.all(f == fold)])
                pipeline.fit(X_train_full[:, train_trials].T, y_dummy[train_trials])
                X_test_full = test_data.transpose((1, 0, 2)).reshape(n_cells, n_trial_types * max_n_trials)
                #             nan_inds = np.isnan(X_train_full)
                scores[i_fold] = pipeline.score(X_test_full[:, fold].T, y_dummy[fold])
                if kern == 'linear' and n_trial_types == 2:
                    coefs[i_fold] = pipeline.named_steps['estimator'].coef_
            score = np.mean(scores)

    else:
        raise ValueError('Data must be either 5 or 4-dimensional')

    return score, coefs, confusion, correct


def make_filename(subc, method):
    name = '_'.join([subc, method])
    filepath = os.path.join(os.path.dirname(os.getcwd()), 'pop_code_fits', name)
    filename = os.path.join(filepath, name + '.p')
    return filepath, filename


def parse_tsv(ret, neuron_info, inc_cells, kim_atlas=None):
    """
    :param ret: sqlite3.Row in the ephys table
    :param neuron_info: dictionary of lists
    :param inc_cells: boolean array (from load_processed.py) OR array containing a single int (from regress_cell.py)
    :return:
    """
    if len(inc_cells) == 1:
        n_cells = 1
    else:
        n_cells = np.sum(inc_cells)
    paths = get_db_info()
    data_path = ret['processed_data_path'].replace(paths['remote_ephys_root'], paths['ephys_root'])
    df = pd.read_csv(os.path.join(data_path, 'cluster_info.tsv'), delimiter='\t')
    good_inds = df['group'] == 'good'
    # get the channel locations, either registered or approximate
    loc_dir = Path(data_path).parent.absolute()

    try:
        chn_coords = np.load(os.path.join(loc_dir, 'channels.localCoordinates.npy'))
        chn_x = np.unique(chn_coords[:, 0])
        chn_x_diff = np.diff(chn_x)
        n_shanks = np.sum(chn_x_diff > 100) + 1

    except FileNotFoundError:
        n_shanks = 1

    if n_shanks == 1:
        loc_files = [os.path.join(loc_dir, 'channel_locations.json')]
    else:
        loc_files = [os.path.join(loc_dir, 'channel_locations_shank{}.json'.format(i)) for i in range(1, n_shanks+1)]

    if np.all([os.path.isfile(loc_file) for loc_file in loc_files]):

        locs = []
        for loc_file in loc_files:
            with open(loc_file, 'rb') as f:
                locs.append(json.load(f))
        # all_ccf_coords = np.empty(shape=(0, 3))
        for ch in df['ch'][good_inds].iloc[inc_cells]:

            if n_shanks == 1:
                loc = locs[0]['channel_' + str(ch)]
            else:
                # this_chn_x = chn_coords[ch, 0]
                this_shank = np.flatnonzero(chn_x == chn_coords[ch, 0])[0] // 2
                these_xs = chn_x[this_shank*2:(this_shank+1)*2]
                chn_ind_within_shank = np.sum(np.flatnonzero(np.isin(chn_coords[:, 0], these_xs)) < ch)
                loc = locs[this_shank]['channel_' + str(chn_ind_within_shank)]

            neuron_info['depths'].append(loc['z'] / 1e3)
            neuron_info['aps'].append(loc['y'] / 1e3)
            # neuron_info['mls'].append(np.abs(loc['x'] / 1e3))
            # get the correct hemisphere on the basis of ephys table, not whatever ibl atlaselectrophys thinks
            if ret['probe1_ML'] >= 0:
                neuron_info['mls'].append(np.abs(loc['x'] / 1e3))
            else:
                neuron_info['mls'].append(-np.abs(loc['x'] / 1e3))
            neuron_info['regions'].append(loc['brain_region'])
            neuron_info['region_ids'].append(loc['brain_region_id'])

            # transform into Allen CCF coordinates
            # bregma is hardcoded based on IBL convention
            if kim_atlas is not None:
                ccf_coords = np.array([5400, 332, 5739]) - np.array([loc['y'], loc['z'], loc['x']])
                neuron_info['kim_regions'].append(kim_atlas.structure_from_coords(ccf_coords, as_acronym=True, microns=True))
                neuron_info['kim_region_ids'].append(int(kim_atlas.structure_from_coords(ccf_coords, microns=True)))
                try:
                    neuron_info['kim_generals'].append(kim_atlas.structure_from_coords(ccf_coords, as_acronym=True, microns=True, hierarchy_lev=6))
                except IndexError:
                    neuron_info['kim_generals'].append(
                        kim_atlas.structure_from_coords(ccf_coords, as_acronym=True, microns=True, hierarchy_lev=-1))
    else:
        depths = df[good_inds]['depth']
        neuron_info['depths'].extend(-ret['probe1_DV'] + depths[inc_cells] / 1e3)
        neuron_info['aps'].extend([ret['probe1_AP']] * n_cells)
        # neuron_info['mls'].extend([np.abs(ret['probe1_ML'])] * n_cells)
        neuron_info['mls'].extend([ret['probe1_ML']] * n_cells)
        for key in ['regions', 'region_ids', 'kim_regions', 'kim_region_ids', 'kim_generals']:
            neuron_info[key].extend([None] * n_cells)

    return neuron_info, df[good_inds]

def poisson_deviance(y, mu):
    # formula from https://en.wikipedia.org/wiki/Deviance_(statistics)
    # and https://thestatsgeek.com/2014/04/26/deviance-goodness-of-fit-test-for-poisson-regression/
    mu = mu.copy()
    if isinstance(mu, float):
        mu = np.repeat(mu, len(y))
    # handle case where y = 0. See
    # https://stats.stackexchange.com/questions/15730/poisson-deviance-and-what-about-zero-observed-values
    mask = y != 0
    nzero = y[mask] * np.log(y[mask] / mu[mask]) - y[mask] + mu[mask]
    return 2 * (np.sum(nzero) + np.sum(mu[~mask]))


def fit_gmm(arr):
    lowest_bic = np.infty
    bic = []
    gmms = []
    n_gmm_components_range = range(1, 10)
    for n_gmm_components in n_gmm_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_gmm_components, covariance_type='full', max_iter=1000, random_state=1)
        gmm.fit(arr)
        bic.append(gmm.bic(arr))
        gmms.append(gmm)

    # Plot the BIC scores
    plt.figure()
    plt.plot(n_gmm_components_range, bic, 'o-', lw=2)
    plt.xticks(n_gmm_components_range)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    hide_spines()

    return gmms


def plot_gmm_preds(arr, gmms, best_gmm_n_components):
    best_gmm_ind = best_gmm_n_components - 1
    clf = gmms[best_gmm_ind]  # use the one with 4 components, which has index 3
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])

    best_gmm_labels = clf.predict(arr)
    red_array = PCA(n_components=2).fit_transform(arr)
    fig, ax = plt.subplots()
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
        v, w = linalg.eigh(cov)
        plt.scatter(red_array[best_gmm_labels == i, 0], red_array[best_gmm_labels == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(.5)
        ax.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(f'Selected GMM: {clf.covariance_type} model, {clf.n_components} components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    hide_spines()

    print(np.unique(best_gmm_labels, return_counts=True))

    return best_gmm_labels


def spectral_clust(arr):
    n_spectral_components = range(2, 5)
    n_nearest_neighbors = range(4, 10)
    n_cells = arr.shape[0]
    all_labels = np.zeros((len(n_spectral_components), len(n_nearest_neighbors), n_cells))
    silhouettes = np.zeros((len(n_spectral_components), len(n_nearest_neighbors)))
    for i, i_spect in enumerate(n_spectral_components):
        for j, knn in enumerate(n_nearest_neighbors):
            clustering = SpectralClustering(n_clusters=i_spect, assign_labels='discretize',
                                            random_state=1, affinity='nearest_neighbors',
                                            n_neighbors=knn, n_jobs=-1).fit(arr)
            all_labels[i, j, :] = clustering.labels_
            silhouettes[i, j] = silhouette_score(arr, clustering.labels_)

    plt.figure()
    plt.pcolormesh(np.arange(3.5, 10), np.arange(1.5, 5), silhouettes)
    plt.xlabel('K nearest neighbors')
    plt.ylabel('Number of clusters')
    plt.title('Silhouette coefficient')
    plt.ylim(max(n_spectral_components) + .5, 1.5)
    plt.yticks(n_spectral_components)
    plt.colorbar()

    best_ind = np.unravel_index(np.argmax(silhouettes), silhouettes.shape)
    best_spectral_labels = all_labels[best_ind]
    best_spectral_n_components = n_spectral_components[best_ind[0]]
    print(np.unique(best_spectral_labels, return_counts=True))

    return best_spectral_labels, best_spectral_n_components


def distplot(contribs, groups, clabels, name, box=True):
    n_clust = len(np.unique(clabels))
    fig, axs = plt.subplots(1, n_clust, figsize=(n_clust*5, 6), squeeze=False)
    for i_clust in range(n_clust):
        ax = axs[0, i_clust]
        if box:
            sns.boxplot(data=contribs[clabels == i_clust], ax=ax)
        else:
            sns.violinplot(data=contribs[clabels == i_clust], ax=ax)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=90, size=12)
        if i_clust == 0:
            ax.set_ylabel('Relative contribution')
        if n_clust > 1:
            ax.set_title('Cluster {}'.format(i_clust+1))
    plt.suptitle(name)
    fig.tight_layout()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    return spatial.distance.cosine(v1, v2)


def assign_str_regions_from_kim(neuron_info, kim_key='kim_regions'):
    """
    :param neuron_info: dictionary or DataFrame
    :param kim_key: key used to index neuron_info
    :return: dictionary or DataFrame with 'str_regions' added as a key, grouping kim_key
    """
    regions = {'OT': ['Tu1', 'Tu2', 'Tu3'],
               'VP': 'VP',
               'mAcbSh': 'AcbSh',
               'lAcbSh': ['LAcbSh', 'CB', 'IPACL'],
               'core': 'AcbC',
               'VMS': ['CPr, imv', 'CPi, vm, vm', 'CPi, vm, v', 'CPi, vm, cvm'],
               'VLS': ['CPr, l, vm', 'CPi, vl, imv', 'CPi, vl, v', 'CPi, vl, vt', 'CPi, vl, cvl'],
               'DMS': ['CPr, m', 'CPr, imd', 'CPi, dm, dl', 'CPi, dm, im', 'CPi, dm, cd', 'CPi, dm, dt'],
               'DLS': ['CPr, l, ls', 'CPi, dl, d', 'CPi, dl, imd'],
               }

    if neuron_info is None:
        return regions
    else:
        neuron_info['str_regions'] = neuron_info[kim_key].copy()
        for k, v in zip(regions.keys(), regions.values()):
            if type(v) == list:
                val = np.array([np.any([subreg == x for subreg in v]) for x in neuron_info[kim_key]])
            else:
                val = np.array([v == x for x in neuron_info[kim_key]])
            neuron_info['str_regions'][val] = k

        return pd.DataFrame(neuron_info), list(regions.keys())


def construct_where_str(protocol, kw, table='imaging'):

    # create SQL query based on keyword arguments passed to function
    where_str = ['protocol="' + protocol.replace('DiverseDists', 'DistributionalRL_6Odours') + '" AND exclude=0 AND has_' + table + '=1']
    # where_vals = []

    if kw['manipulation'] is None and table == 'ephys':  # gets its own special case because here we want to do something with the None value
        if 'additional_names' in kw:
            where_str.append('(' + table + '.name IN(SELECT name FROM mouse WHERE surgery1="headplate") OR (' + table + \
                         '.name IN ' + str(kw['additional_names']) + ' AND session.probe1_ML < 0))')
        else:
            where_str.append('(' + table + '.name IN(SELECT name FROM mouse WHERE surgery1="headplate"))')
    elif kw['manipulation'] == 'combined':
        where_str.append('(' + table + '.name IN(SELECT name FROM mouse WHERE surgery1="headplate") OR ' + table + '.probe1_ML < 0)')
    elif table == 'ephys':
        where_str.append('(' + table + '.name IN(SELECT name FROM mouse WHERE surgery1 LIKE "%' + kw['manipulation'] + '%") AND NOT ' + \
                         table + '.name IN ' + str(kw['additional_names']) + ')')

    for key, val in zip(kw.keys(), kw.values()):
        if val is not None:
            if table == 'imaging':
                if key == 'code':
                    where_str.append(table + '.name IN(SELECT name FROM mouse WHERE code IN ' + str(val) + ')')
                elif key == 'genotype':
                    where_str.append(table + '.name IN(SELECT name FROM mouse WHERE genotype LIKE "%' + str(val) + '%")')
                elif key == 'continuous':
                    where_str.append(key + '=' + str(val))
                    # where_vals.append(val)
                elif key == 'wavelength':
                    where_str.append(key + '<=' + str(val))
            elif table == 'ephys' and key == 'probe1_region':
                where_str.append(key + '=' + '"' + str(val) + '"')

            if key in ['n_trial', 'quality', 'phase']:
                where_str.append(key + '>=' + str(val))
                # where_vals.append(val)
            elif key == 'significance':
                where_str.append('session.' + key + '=' + str(val))
                # where_vals.append(val)
            elif key == 'curated':
                where_str.append(key + '=' + str(val))
                # where_vals.append(val)
            elif key == 'name':
                where_str.append(table + '.name IN ' + str(val))
            elif key == 'exclude_names':
                where_str.append(table + '.name NOT IN ' + str(val))
            elif key == 'exclude_sess':
                where_str.append('NOT (' + ' OR '.join([f'({table}.name="{x}" AND {table}.file_date={y})' for x, y, in val]) + ')')

    where_str = ' AND '.join(where_str)

    cols = ['session.name', 'session.exp_date', table + '.figure_path', 'behavior_path', 'file_date_id',
            table + '.file_date', table + '.processed_data_path', table + '.meta_time', 'ncells', 'stats',
            'session.mid', 'sid', 'rid', 'session.exp_date', 'session.probe1_AP', 'session.probe1_ML',
            'session.probe1_DV', 'session.significance', 'mouse.genotype', 'n_trials_used']
    if table == 'ephys': cols.append('n_trials_used')

    sql = 'SELECT ' + ', '.join(cols) + ' FROM ' + table + ' LEFT JOIN session ON ' + table + '.behavior_path = ' + \
          'session.raw_data_path LEFT JOIN mouse ON ' + table + '.name = mouse.name WHERE ' + where_str + \
          ' ORDER BY session.mid ASC, ' + table + '.file_date ASC'

    return where_str, sql


def plot_most_sign(diff, times, timecourses, colors, n_trace_types, n_plot=20, mini=-1):
    """
    plot neurons with largest differences
    :param diff: The differences computed on the metric of interest (e.g. auROC)
    :param times: Timebase for timecourses
    :param timecourses: Dictionary with cs, rew, and combo responses
    :param n_plot: Number of cells to plot
    :param mini: Should be either 1 or -1. If -1, use the maximum differences (in the direction of the predicted effect.
    If 1, use the minimum differences (generally in the opposite direction of the predicted effect). Default is -1.
    :return:
    """
    max_diff_inds = np.argpartition(diff, mini*n_plot)[mini*n_plot:]
    max_diff_inds = max_diff_inds[np.argsort(diff[max_diff_inds])][::mini]
    n_rows = 4
    fig, axs = plt.subplots(n_rows, n_plot // n_rows, figsize=(12, n_rows * 2))
    trace_dict = {'cs_in': 0,
                  'cs_out': 1,
                  'trace_end': 3,
                  'xlim': (-1, 5),
                  'ylabel': 'z-scored FR',
                  'xlabel': 'Time from CS (s)'
                  }
    for i, diff_ind in enumerate(max_diff_inds):
        ax = axs.flat[i]
        ax.set_prop_cycle(color=colors['colors'])
        ax.plot(times, timecourses['cs']['zF'][:n_trace_types, diff_ind, :].T)
        setUpLickingTrace(trace_dict, ax=ax, override_ylims=True)
    hide_spines()
    fig.tight_layout()


def my_z(arr, axis):
    tmp = stats.zscore(arr, axis=axis)
    tmp[np.isnan(tmp)] = 0
    return tmp

