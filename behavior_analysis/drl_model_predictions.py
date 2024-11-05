import h5py
import os
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcol
import pickle
import sys
from scipy import stats
from statsmodels.formula.api import mixedlm
import cmocean
from copy import deepcopy
from itertools import permutations, product
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from traceUtils import *

sys.path.append('../utils')
from expectiles import infer_dist, run_decoding
from plotting import hide_spines, plot_stars, set_share_axes
from plotting import SeabornFig2Grid as sfg
from db import get_db_info, select_db
from protocols import load_params

sys.path.append('../behavior_analysis')
from traceUtils import *


def get_mean_reflected(preds, half_preds, clamp_val):
    preds[:half_preds] = clamp_val - preds[:half_preds]
    return np.mean(preds), preds


def generate_model_predictions(protocol_info, colors, cat=True, n_components=2, noise_scale=5, plot_detail=False, subsets=['all']):
    """
    :param protocol_info:
    :param colors:
    :param cat:
    :param n_components:
    :param noise_scale:
    :param plot_detail:
    :param subsets: should be either ['all'] or ['all', 'pess', 'opt], in that order
    :return:
    """

    protocol = protocol_info['protocol']
    n_dup = 2 if 'Same' in protocol else 1
    # n_unique_dists = protocol_info['n_trace_types'] / n_dup

    n_preds = 10
    half_preds = int(n_preds / 2)

    max_val = np.amax([val for dist in protocol_info['dists'] for val in dist])
    bin_width = np.amin(np.diff(np.unique([val for dist in protocol_info['dists'] for val in dist])))  # 1  # 2
    if bin_width == 6: bin_width = 2
    print(bin_width)
    # bin_edges = np.arange(-1, max_val + 2, 2)
    bin_edges = np.arange(0 - bin_width / 2, max_val + bin_width, bin_width)
    print(bin_edges)
    histos = [np.histogram(x, bin_edges, density=True)[0] * bin_width for x in protocol_info['dists'][:protocol_info['n_trace_types']:n_dup]]

    floor = 0
    # if cat:
    #     for histo in histos:
    #         histo[histo >= floor * 2] -= floor * np.sum(histo < floor) / np.sum(histo >= floor * 2)
    #         histo[histo < floor] = floor
    ceiling = np.amax([np.amax(x) for x in histos])
    print(ceiling)

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    reward_values = bin_centers
    nb = len(bin_centers)
    print(reward_values)

    # bin_centers = np.array([0, 2, 4, 6, 8])
    # reward_values = np.array([0, 2, 4, 6, 8])

    # histos = [np.array([.85, .05, .05, .05]), np.array([.05, .05, .85, .05]), np.array([.05, .45, .05, .45])]

    # histos = [np.array([.85, 0, .05, 0, .05, 0, .05]),
    #           np.array([.05, 0, .05, 0, .85, 0, .05]),
    #           np.array([.05, 0, .45, 0, .05, 0, .45])]
    # histos = [np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    #           np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    #           np.array([0, 0, 0, 0, .5, 0, 0, 0, .5, 0, 0])]

    cum_histos = [np.cumsum(x) for x in histos]
    rng = np.random.default_rng(seed=1)

    # bin_centers = np.arange(-2, 9)
    # reward_values = np.arange(-2, 9)
    n_trials = 10000
    print(histos)
    distros = [rng.choice(reward_values, size=n_trials, p=histo) for histo in histos]

    taus = np.linspace(.05, .95, n_preds)
    print(taus)
    grey = [.5] * 3
    clamp_val = max_val
    # clamp_val = 0
    n_rows = 3
    # distro_names = ['Nothing', 'Fixed', 'Variable']
    distro_names = [x.split()[0] for x in protocol_info['trace_type_names'][::n_dup]] if n_dup > 1 else protocol_info['trace_type_names']
    n_dists = len(distro_names)

    pess_color = '#BA55D3'
    opt_color = '#0047AB'

    info = {'reward_values': reward_values,
        'distro_names': distro_names,
        'pess_color': pess_color,
        'opt_color': opt_color,
        'distros': distros}
    df_list = []  # will get overwritten

    pess_neurons = opt_neurons = None

    if cat:

        # used for categorical, cumulative, and laplace codes to linearly map histograms to neural activity
        # tuning_curves = np.zeros((nb + 1, nb))
        # tuning_curves[0, 0] = 1
        # tuning_curves[-1, -1] = 1
        # for i_tc, tc in enumerate(tuning_curves[1:-1]):
        #     tc[i_tc:i_tc + 2] = .5
        # tuning_curves = tuning_curves.T
        tuning_curves = np.eye(nb)

        pess_neurons = np.arange(int(np.floor(nb / 2))).astype(np.int32)
        print(pess_neurons)
        # opt_neurons = np.arange(int(np.ceil((nb + 1) / 2)), nb + 1).astype(np.int32)
        opt_neurons = np.arange(int(np.ceil(nb / 2)), nb).astype(np.int32)
        print(opt_neurons)
        # tuning_curves = np.array([[1, 0, 0, 0],
        #                          [.5, .5, 0, 0],
        #                          [0, .5, .5, 0],
        #                          [0, 0, .5, .5],
        #                          [0, 0, 0, 1]]).T
        # pess_neurons = np.array([0, 1])
        # opt_neurons = np.array([3, 4])
        manip_neurons = [pess_neurons, opt_neurons]
        manip_bins = [np.array([0, 1]), np.array([-2, -1])]

    # choose basis functions for categorical, cumulative, and laplace coding
    n_trace_types = protocol_info['n_trace_types']
    n_sim_per_type = 50
    n_reps_per_cell = 100
    crit = .05
    ms = 1  # marker size

    odor_activity = np.unique(list(permutations([1, 0, 0, 0, 0, 0], r=n_trace_types)), axis=0)

    # sandra and mikhael params
    alpha = .2
    beta = .01
    # # opal parameters
    # opal_alpha_c = .2
    # opal_alpha = .3

    redrl_alpha_avg = .015
    redrl_beta = 4  # relative boost given to positive (D1) or negative (D2) RPEs

    alpha_ps, alpha_ns = get_alphas(taus, redrl_alpha_avg)
    alpha_ps_d1 = alpha_ps * redrl_beta
    alpha_ns_d2 = alpha_ns * redrl_beta
    tau_star_d1 = alpha_ps_d1 / (alpha_ps_d1 + alpha_ns)
    tau_star_d2 = alpha_ps / (alpha_ps + alpha_ns_d2)
    tau_star = np.stack((tau_star_d1, tau_star_d2), axis=0)
    print(tau_star)

    otaus = taus.copy()
    # uncomment to use different distribution of taus for (R)E/QDRL
    # taus = tau_star[np.array([1, 0])].flatten()  # get d2 taus first, for when indexing by half_preds
    # half_preds = len(taus) // 2

    if cat:
        # generate hypothetical activity for each code type
        cat_activity = np.repeat(np.vstack(histos), n_dup, axis=0) @ tuning_curves  # * 10  # times just for scaling
        cum_activity = np.repeat(np.vstack(cum_histos), n_dup, axis=0) @ tuning_curves  # * 10
        lap_activity = np.repeat(1 - np.vstack(cum_histos), n_dup, axis=0) @ tuning_curves  # * 10
    q_activity = np.vstack([np.quantile(distro, taus) for distro in np.repeat(distros, n_dup, axis=0)])
    e_activity = np.vstack([infer_dist(taus=taus, dist=distro) for distro in np.repeat(distros, n_dup, axis=0)])
    # e_activity = np.vstack([stats.expectile(taus=taus, dist=distro) for distro in np.repeat(distros, 2, axis=0)])
    rq_activity = q_activity.copy()
    rq_activity[:, :half_preds] = clamp_val - rq_activity[:, :half_preds]
    re_activity = e_activity.copy()
    re_activity[:, :half_preds] = clamp_val - re_activity[:, :half_preds]
    moment_activity = np.vstack([[np.mean(distro), np.var(distro)] for distro in np.repeat(distros, n_dup, axis=0)])

    # for John Mikhael's/Sandra's extended model/OpAL, need to actually simulate the outcomes
    sandra_activity = np.zeros((2, n_dists, n_preds))
    mikhael_activity = np.zeros((2, n_dists))
    # opal_v = np.zeros(n_dists)
    # opal_activity = np.ones((2, n_dists))

    # also do this for redrl for completeness
    redrl_activity = np.zeros((2, n_dists, n_preds))
    rqdrl_activity = np.zeros((2, n_dists, n_preds))
    rdrl_update = np.zeros((2, n_preds))


    for i_dist, distro in enumerate(distros):
        # print(distro, distro.shape)
        # plt.figure()
        # plt.hist(distro)
        # plt.show()
        for i_trial in range(n_trials):

            # sandra
            V = sandra_activity[0, i_dist, :] - sandra_activity[1, i_dist, :]

            # print(distro[i_trial])
            delta = distro[i_trial] - V
            isneg = np.array(delta < 0, dtype=np.float32)

            # print(sandra_activity[0, i_dist, :].shape, taus.shape, isneg.shape, delta.shape)
            # update = alpha * np.abs(taus - isneg) * delta * ~isneg.astype(bool) - beta * sandra_activity[0, i_dist, :]
            # print(delta, update)
            # could revert to taus instead of otaus if not using extremized taus from tau star
            sandra_activity[0, i_dist, :] += alpha * np.abs(otaus - isneg) * np.abs(delta) * ~isneg.astype(bool) - beta * sandra_activity[0, i_dist, :]
            sandra_activity[1, i_dist, :] += alpha * np.abs(otaus - isneg) * np.abs(delta) * isneg.astype(bool) - beta * sandra_activity[1, i_dist, :]

            # mikhael
            Vmean = mikhael_activity[0, i_dist] - mikhael_activity[1, i_dist]
            delta_mean = distro[i_trial] - Vmean
            isneg_mean = delta_mean < 0

            mikhael_activity[0, i_dist] += alpha * np.abs(delta_mean) * ~isneg_mean.astype(bool) / 2 - beta * mikhael_activity[0, i_dist]
            mikhael_activity[1, i_dist] += alpha * np.abs(delta_mean) * isneg_mean.astype(bool) / 2 - beta * mikhael_activity[1, i_dist]

            for activity, scale, fun in zip([redrl_activity, rqdrl_activity], [1, 4], [lambda x: x, np.sign]):
                Vis = activity.copy()
                Vis[1, :, :] = clamp_val - activity[1, :, :]  # D2 -> ventral pallidum
                delta = distro[i_trial] - Vis[:, i_dist, :]  # dopamine
                isneg = np.array(delta < 0, dtype=np.float32)
                rdrl_update[0] = fun(delta[0]) * scale * (isneg[0] * alpha_ns + (1 - isneg[0]) * alpha_ps_d1)  # compute update
                rdrl_update[1] = -fun(delta[1]) * scale * (isneg[1] * alpha_ns_d2 + (1 - isneg[1]) * alpha_ps)
                activity[:, i_dist, :] += rdrl_update  # update striatum

            # # opal
            # opal_delta = distro[i_trial] - opal_v[i_dist]
            #
            # # update critic
            # opal_v += opal_alpha_c * opal_delta  # Eq. 1
            #
            # # update actor
            # opal_activity[0, i_dist] += opal_alpha * opal_activity[0, i_dist] * opal_delta  # Eq. 2
            # opal_activity[1, i_dist] -= opal_alpha * opal_activity[1, i_dist] * opal_delta  # Eq. 3
            #
            # # constrained to be positive
            # # opal_activity[opal_activity < 0] = 0

    # set order of iteration
    if cat:
        code_order = ['Reflected Expectile', 'Expectile', 'Quantile', 'Reflected Quantile', 'Distributed AU',
                      'Partial Distributed AU', 'Actor Uncertainty (AU)', 'Categorical', 'Laplace', 'Cumulative', 'Moments']
        activities = [e_activity, re_activity, q_activity, rq_activity, sandra_activity, sandra_activity, mikhael_activity,
                      cat_activity, lap_activity, cum_activity, moment_activity]
    else:
        code_order = ['Reflected Expectile', 'Expectile', 'Quantile', 'Reflected Quantile', 'Distributed AU',
                      'Partial Distributed AU', 'Actor Uncertainty (AU)', 'Moments']
        activities = [e_activity, re_activity, q_activity, rq_activity, sandra_activity, sandra_activity, mikhael_activity,
                      moment_activity]

    # subsets = ['all', 'pess', 'opt']
    # subsets = ['all']
    ns = len(subsets)
    code_order_to_pca = code_order  # ['Reflected Expectile', 'Reflected Quantile']
    nc = len(code_order_to_pca)

    pairwise_dists = np.zeros((ns, nc, 2, n_dup ** 2))
    pairwise_dists2 = np.zeros((ns, nc, 2, n_dup ** 2))
    rdas = np.zeros((ns, nc, 2, n_dup ** 2))
    mean_dists = np.zeros((ns, nc, 3 * n_dup ** 2))
    var_dists = np.zeros((ns, nc, 6))

    reduced_activities = np.empty((ns, nc, n_components), dtype='object')
    all_dists = np.zeros((ns, nc, n_components, n_trace_types, n_trace_types))
    all_rda = np.zeros((ns, nc, n_trace_types, n_trace_types))

    for i_grp, (which_neurons, neuron_inds) in enumerate(zip(subsets, [0, pess_neurons, opt_neurons])):
    # for i_grp, which_neurons in enumerate(subsets):

        # pre-allocate figures
        activities_to_pca = activities
        # code_order_to_pca = code_order

        fig, axs = plt.subplots(3, nc, figsize=(2 * nc, 6.5), squeeze=False,
                                gridspec_kw={'wspace': .5, 'hspace': .5})  # , sharex=True, sharey=True)
        fig3, axs3 = plt.subplots(1, nc, figsize=(2 * nc, 3), squeeze=False,
                                  gridspec_kw={'wspace': .5, 'hspace': .5}, subplot_kw={'projection': '3d'})
        # [set_share_axes(axs[i, :], sharey=True) for i in [1, 2]]
        [set_share_axes(axs[i, :], sharey=True) for i in [2]]
        # scatter_fig, scatter_axs = plt.subplots(1, len(code_order), figsize=(2*len(code_order), 2), sharey=True)

        for i_code, (hypothetical_activity, code_name) in enumerate(zip(activities_to_pca, code_order_to_pca)):

            print(code_name)
            stat_dict = {'stat': [], 'slope': [], 'intercept': [], 'Correlation coefficient': [], 'pvals': [], 'shuff': []}
            prc_dict = {'stat': [], 'Percentage': [], 'Type': [], 'shuff': []}
            #     for stat, stat_name in zip([protocol_info['mean'][:n_trace_types], protocol_info['var'][:n_trace_types],
            #                                 *np.unique(list(permutations([1, 0, 0, 0, 0, 0], r=n_trace_types)), axis=0)],
            #                                ['Mean', 'Variance'] + ['Odor'] * n_trace_types):
            #                                 *list(permutations(protocol_info['mean'][:n_trace_types], r=n_trace_types)),
            #                                 *list(permutations(protocol_info['var'][:n_trace_types], r=n_trace_types))],
            #                                ['Mean', 'Variance'] + ['Shuff Mean'] * 6 + ['Shuff Var'] * 6):

            #         for shuff, activity in enumerate([hypothetical_activity.T, odor_activity]):
            for shuff in range(2):

                if 'AU' in code_name:
                    n_pops = 2
                    activity = np.repeat(hypothetical_activity, n_dup, axis=1)  # duplicate for Sandra and Mikhael codes
                    if 'Distributed' in code_name:
                        use_half_preds = half_preds  # * 2
                        if code_name == 'Distributed AU':
                            # treat neurons as pessimistic if tau < .5
                            activity = activity.transpose(1, 2, 0).reshape((n_dists * n_dup, n_preds * n_pops)).T
                            basis = np.repeat(otaus, n_pops)
                        else:
                            # treat neurons as pessimistic if D2
                            activity = activity.transpose(1, 0, 2).reshape((n_dists * n_dup, n_preds * n_pops)).T
                            # make first rows the D2 (pessimistic) cells
                            activity = activity[np.concatenate((np.arange(n_preds, n_preds * n_pops), np.arange(n_preds)))]
                            basis = np.tile(otaus, n_pops)
                    else:
                        # make first row the D2 (pessimistic) cells
                        activity = activity[np.array([1, 0])]
                        use_half_preds = 1
                        basis = np.array([.5, .5])
                else:
                    activity = hypothetical_activity.T.copy()

                if code_name in ['Categorical', 'Cumulative', 'Laplace']:
                    basis = reward_values
                    if which_neurons != 'all':
                        print(neuron_inds)
                        print(reward_values)
                        print(which_neurons)
                        print(code_name)
                        activity = activity[neuron_inds, :]
                        basis = reward_values[neuron_inds]
                else:
                    if code_name == 'Moments':
                        use_half_preds = 1
                        basis = np.array([.5, .5])
                    elif 'Quantile' in code_name or 'Expectile' in code_name:
                        use_half_preds = half_preds
                        basis = taus

                    if which_neurons == 'pess':
                        activity = activity[:use_half_preds]
                        basis = basis[:use_half_preds]
                    elif which_neurons == 'opt':
                        activity = activity[use_half_preds:]
                        basis = basis[use_half_preds:]

                print(basis)
                # mult = 1 if which_neurons == 'all' else -1
                mult = -1

                n_sim_cells = activity.shape[0]
                noise = rng.normal(scale=np.std(hypothetical_activity.flatten()) * noise_scale,
                                   size=(n_reps_per_cell * n_sim_cells, n_sim_per_type * n_trace_types))

                # print('n_reps_per_cell', n_reps_per_cell)
                # print('n_sim_cells', n_sim_cells)
                # print('n_sim_per_type', n_sim_per_type)
                # print('n_trace_types', n_trace_types)
                # print(noise.shape)
                # print(activity.shape)
                # noise = rng.normal(scale=np.std(activity, axis=1, keepdims=True).repeat(n_reps_per_cell, axis=0) * 5,
                #                    size=(n_reps_per_cell * n_sim_cells, n_sim_per_type * n_trace_types))
                #             print(noise.shape)
                sim_activity = activity.repeat(n_reps_per_cell, axis=0)
                print(sim_activity.shape)
                print(n_sim_per_type)
                print(sim_activity.repeat(n_sim_per_type, axis=1).shape)
                print(n_reps_per_cell, n_sim_cells, n_sim_per_type, n_trace_types)
                print(noise.shape)
                # print(sim_activity.shape)
                if shuff == 1:
                    [rng.shuffle(x) for x in sim_activity]
                sim_activity = sim_activity.repeat(n_sim_per_type, axis=1) + noise
                # print(sim_activity.shape)

                order = ['Mean', 'Variance']  # 'Residual\nVariance']  # Variance

                for stat, stat_name in zip([protocol_info['mean'][:n_trace_types]] +
                                           [protocol_info['var'][:n_trace_types]] * 1, order):

                    #             print(sim_activity.shape)
                    #             print(activity)

                    if stat_name == 'Mean' and shuff == 0:
                        # perform PCA on hypothetical activity to project into two dimensions, then plot in PC space
                        #                 reduced_activity = PCA(n_components=n_components).fit_transform(hypothetical_activity)
                        sim_type_avg = np.stack(
                            [np.mean(sim_activity[:, i_tt * n_sim_per_type:(i_tt + 1) * n_sim_per_type], axis=1)
                             for i_tt in range(n_trace_types)], axis=0)
                        #                 print(sim_type_avg.shape)
                        pca = PCA(n_components=n_components)
                        # pca = FastICA(n_components=n_components)
                        # pca.n_components_ = 2
                        # # print(sim_type_avg.shape)
                        reduced_activity = pca.fit_transform(sim_type_avg)

                        p = np.zeros((n_components, n_trace_types, n_trace_types))

                        # print(sim_type_avg.shape)
                        rda = pairwise_distances(sim_type_avg, metric='cosine')
                        # print(rda.shape)

                        for i_comp in range(n_components):
                            reduced_activities[i_grp, i_code, i_comp] = reduced_activity[:, i_comp]
                            p[i_comp] = pairwise_distances(reduced_activity[:, i_comp].reshape(-1, 1), metric='euclidean')

                        all_dists[i_grp, i_code, :, :, :] = p
                        all_rda[i_grp, i_code, :, :] = rda

                        print(p[0])
                        print(p[1])

                        for imat, (dmat, darr) in enumerate(zip([p[0], p[1], rda], [pairwise_dists, pairwise_dists2, rdas])):
                            if n_dup == 2:
                                nvv = np.array([dmat[x] for x in product([0, 1], [4, 5])])  # nothing vs. variable
                                nvf = np.array([dmat[x] for x in product([0, 1], [2, 3])])  # nothing vs. fixed
                                if imat == 0:  # distances along PC 1
                                    across_mean = np.array([dmat[x] for x in product([0, 1], range(2, 6))])  # nothing vs. fixed or variable
                                    within_mean = np.array([dmat[x] for x in product([2, 3], [4, 5])])  # fixed vs. variable
                                elif imat == 1:  # distances along PC 2
                                    across_dist = np.array([dmat[x] for x in product([2, 3], [4, 5])])  # fixed vs. variable
                                    within_dist = np.array([dmat[2, 3], dmat[4, 5]])  # fixed 1 vs fixed 2; variable 1 vs. variable 2
                            elif n_dup == 1:
                                nvv = np.array([dmat[0, 2]])
                                nvf = np.array([dmat[0, 1]])
                                if imat == 0:
                                    across_mean = np.array([dmat[0, 1], dmat[0, 2]])
                                    within_mean = np.array([dmat[1, 2]])
                            darr[i_grp, i_code, :, :] = np.vstack((nvv, nvf))
                            if imat == 0:
                                mean_dists[i_grp, i_code, :] = np.concatenate((across_mean, within_mean))
                            elif imat == 1 and n_dup == 2:
                                var_dists[i_grp, i_code, :] = np.concatenate((across_dist, within_dist))

                        # for dmat, darr in zip([p[0], p[1], rda], [pairwise_dists, pairwise_dists2, rdas]):
                        #
                        #     if n_dup == 2:
                        #         nvv = np.array([dmat[0, 4], dmat[0, 5], dmat[1, 4], dmat[1, 5]])  # nothing vs. variable
                        #         nvf = np.array([dmat[0, 2], dmat[0, 3], dmat[1, 2], dmat[1, 3]])  # nothing vs. fixed
                        #     elif n_dup == 1:
                        #         nvv = np.array([dmat[0, 2]])
                        #         nvf = np.array([dmat[0, 1]])
                        #     darr[i_grp, i_code, :, :] = np.vstack((nvv, nvf))

                        # print(pairwise_dists)
                        # print(rdas)

                        # print(reduced_activity.shape)
                        pca_df = pd.DataFrame({'x': reduced_activity[:, 0], 'y': mult * reduced_activity[:, 1],
                                               'tt': protocol_info['trace_type_names']})
                        ax = axs[0, i_code]
                        sns.scatterplot(data=pca_df, x='x', y='y', hue='tt', hue_order=protocol_info['trace_type_names'],
                                        palette=list(colors['colors'][:n_trace_types]), s=100, ax=ax)
                        #     ax.scatter(reduced_activity[:, 0], reduced_activity[:, 1], s=100, c=colors['colors'][:6:2])
                        ax.set_title(code_name)
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.set_xlim(np.array(ax.get_xlim()) * 1.1)
                        ax.set_ylim(np.array(ax.get_ylim()) * 1.1)
                        # ax.set_ylim([-30, 30])
                        ax.legend().remove()

                        if n_components > 2:
                            ax = axs3[i_code]
                            ax.scatter(reduced_activity[:, 0], reduced_activity[:, 1], reduced_activity[:, 2], color=colors['colors'][:n_trace_types], s=100)

                        ax = axs[1, i_code]
                        print(pca.components_.shape)
                        # ax.plot(np.arange(len(basis) * n_reps_per_cell), pca.components_.T)
                        avg_comps = np.stack([np.mean(pca.components_[:, i_tau * n_reps_per_cell : (i_tau + 1) * n_reps_per_cell], axis=1)
                                              for i_tau in range(len(basis))], axis=0)
                        print(avg_comps.shape, basis.shape)
                        [ax.scatter(basis, avg_comps[:, i_comp]) for i_comp in range(n_components)]

                    if stat_name == 'Residual\nVariance':
                        print('computing resid var')
                        rep_stat = np.array(protocol_info['mean'][:n_trace_types]).repeat(n_sim_per_type)
                        linreg_arr = np.array([stats.linregress(row, rep_stat) for row in sim_activity])
                        # project out the value regression prediction
                        mean_contribution = linreg_arr[:, 1, np.newaxis] + linreg_arr[:, 0, np.newaxis] * rep_stat
                        #                 print(mean_contribution.shape)
                        sim_activity -= mean_contribution
                    #                 print(sim_activity.shape)

                    rep_stat = np.array(stat).repeat(n_sim_per_type)
                    #             print(rep_stat.shape)
                    linreg_arr = np.array([stats.linregress(row, rep_stat) for row in sim_activity])
                    for idx, key in enumerate(['slope', 'intercept', 'Correlation coefficient', 'pvals']):
                        stat_dict[key].extend(linreg_arr[:, idx])

                    stat_dict['stat'].extend([stat_name] * n_sim_cells * n_reps_per_cell)
                    stat_dict['shuff'].extend([shuff] * n_sim_cells * n_reps_per_cell)

                    rvals = linreg_arr[:, 2]
                    pvals = linreg_arr[:, 3]

                    #         if code_name == 'Categorical': print(rvals)
                    #             prc_dict['Percentage'].extend(np.array([np.mean(rvals > epsilon), np.mean(rvals < -epsilon)]) * 100)
                    prc_dict['Percentage'].extend(np.array([np.mean(np.logical_and(pvals < crit, rvals > 0)),
                                                            np.mean(np.logical_and(pvals < crit, rvals < 0))]) * 100)
                    prc_dict['stat'].extend([stat_name] * 2)
                    prc_dict['Type'].extend(['Positive', 'Negative'])
                    prc_dict['shuff'].extend([shuff] * 2)

            ax = axs[2, i_code]
            prc_df = pd.DataFrame(prc_dict).groupby(['stat', 'Type', 'shuff']).mean().reset_index()
            #     .sort_values(by='shuff').reset_index()
            #         by='Type', key=lambda series: [['Negative', 'Positive'].index(x) for x in series]).reset_index()
            #     if code_name == 'Categorical': print(prc_df)
            #     odor_prc = np.repeat(prc_df.loc[prc_df['stat'] == 'Odor', 'Percentage'], 3).values

            #     prc_df.loc[prc_df['stat'] == 'Mean', 'Percentage'] -= prc_df.loc[prc_df['stat'] == 'Shuff Mean', 'Percentage'].values
            #     prc_df.loc[prc_df['stat'] == 'Variance', 'Percentage'] -= prc_df.loc[prc_df['stat'] == 'Shuff Var', 'Percentage'].values
            #     prc_df = prc_df[np.isin(prc_df['stat'], ['Mean', 'Var'])]

            prc_df['Percentage'] -= np.repeat(prc_df.loc[prc_df['shuff'] == 1, 'Percentage'], 2).values
            #     prc_df = prc_df[prc_df['stat'] != 'Odor']
            #     if code_name == 'Categorical': print(prc_df)
            #     prc_df = prc_df[prc_df['shuff'] == 0]
            #     if code_name == 'Categorical': print(prc_df)

            sns.swarmplot(data=prc_df[prc_df['shuff'] == 0], x='stat', y='Percentage', hue='Type', order=order,
                          dodge=True, hue_order=['Positive', 'Negative'], palette=[opt_color, pess_color], size=10, ax=ax)
            #     ax.set_title(code_name)
            ax.set_xticklabels(order, rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlabel('')
            ax.legend().remove()
            ax.axhline(y=0, c=[.5,.5,.5], ls='--')

            stat_df = pd.DataFrame(stat_dict)

            if code_name == 'Categorical':
                plt.figure()
                sns.stripplot(data=stat_df[stat_df['shuff'] == 0], x='stat', y='Correlation coefficient', size=ms)

            axs[0, len(code_order_to_pca) // 2].set_xlabel('Projection onto PC1')
            axs[0, 0].set_ylabel('Projection onto PC2')
            axs[0, -1].legend(loc=(1.04, 0))

            axs[1, len(code_order_to_pca) // 2].set_xlabel(r'$\tau$')
            axs[1, 0].set_ylabel('Mean loadings')
            axs[1, -1].legend(labels=['PC1', 'PC2'], loc=(1.04, 0))

            axs[2, 0].set_ylabel('Sig. cells:\nCode $-$ Odor (%)')
            axs[2, 0].set_ylim([-30, 85])
            axs[2, -1].legend(loc=(1.04, 0))

            # axs[3, 0].set_ylabel('Correlation coefficients')

            # for ax in pca_axs:
            #     ax.set_xlim(np.array(ax.get_xlim())*1.1)
            #     ax.set_ylim(np.array(ax.get_ylim())*1.05)
            # for ax in axs[1, :]:
            #     ax.set_ylim(np.array(ax.get_ylim()) * 1.02)

            fig.tight_layout()
            hide_spines()
            fig.savefig(f'figs/{protocol}_model_comparison_pca_sig_frac_{which_neurons}.png', dpi=300, bbox_inches='tight')
            fig.savefig(f'figs/{protocol}_model_comparison_pca_sig_frac_{which_neurons}.pdf', bbox_inches='tight')

    if protocol == 'DistributionalRL_6Odours' or protocol == 'Bernoulli' or protocol == 'SameRewSize':
        return df_list, code_order, info, reduced_activities, all_dists, all_rda

    elif protocol == 'SameRewDist' or protocol == 'StimGradient':
        # pc1_groupings = ['Nothing vs. Variable', 'Nothing vs. Fixed']
        pc1_groupings = ['N vs. V', 'N vs. F']
        prot_palette = ['#1f77b4', '#d62728']

    elif protocol == 'SameRewVar':
        pc1_groupings = ['Nothing vs. Bimodal', 'Nothing vs. Uniform']
        prot_palette = ['#1f77b4', '#bb4513']

    elif 'Skewness' in protocol:
        pc1_groupings = ['Nothing vs. Skewed', 'Nothing vs. Fixed']
        prot_palette = ['#1f77b4', '#bb4513']

    ngrps = len(pc1_groupings)

    print(pairwise_dists.shape, pairwise_dists2.shape, rdas.shape)
    print(ngrps, n_dup ** 2, ns)
    pdict = {'distance': pairwise_dists.flatten(),
             'distance2': pairwise_dists2.flatten(),
             'rda': rdas.flatten(),
             'code_name': np.tile(np.repeat(code_order_to_pca, ngrps * n_dup ** 2), ns),
             'grouping': np.tile(np.repeat(pc1_groupings, n_dup ** 2), len(code_order_to_pca) * ns),
             'subset': np.repeat(subsets, ngrps * n_dup ** 2 * len(code_order_to_pca))}

    mdict = {'distance': mean_dists.flatten(),
             'code_name': np.tile(np.repeat(code_order_to_pca, 3 * n_dup ** 2), ns),
             'grouping': np.tile(np.repeat(['Across mean', 'Across mean', 'Within mean'], n_dup ** 2), len(code_order_to_pca) * ns),
             'subset': np.repeat(subsets, 3 * n_dup ** 2 * len(code_order_to_pca))}

    vdict = {'distance2': var_dists.flatten(),
             'code_name': np.tile(np.repeat(code_order_to_pca, 3 * 2), ns),
             'grouping': np.tile(np.repeat(['Across dist.', 'Across dist.', 'Within dist.'], 2), len(code_order_to_pca) * ns),
             'subset': np.repeat(subsets, 3 * 2 * len(code_order_to_pca))}

    pdf = pd.DataFrame(pdict)
    mdf = pd.DataFrame(mdict)
    vdf = pd.DataFrame(vdict)
    cdf = pd.concat((mdf, vdf))
    for use_df, depvars, figtit, pals in zip([pdf, cdf], [['distance', 'distance2', 'rda'], ['distance', 'distance2']],
                                             ['nvv_nvf', 'across_within'], [[prot_palette]*3, [['#C4AEAD', '#17BECF'], ['#74B72E', '#FF6600']]]):
        all_subset = use_df[use_df['subset'] == 'all']
        optopess_subset = use_df[use_df['subset'] != 'all']
        for subset_df, col_order in zip([all_subset, optopess_subset], [['all'], ['opt', 'pess']]):
            for depvar, pal, sharey, ylab in zip(depvars, pals, ['row', 'row', False],
                                            ['Distance\nalong\nPC 1 (a.u.)', 'Distance\nalong\nPC 2 (a.u.)', 'Cosine\ndistance']):
                dep_df = subset_df[~np.isnan(subset_df[depvar])]
                if all(x in subsets for x in col_order):
                    g = sns.catplot(dep_df, x='grouping', y=depvar, hue='grouping', row='code_name', col='subset', kind='point',
                                    palette=pal, row_order=code_order_to_pca, col_order=col_order, sharey=sharey,
                                    errwidth=4, height=2.4, aspect=.85, scale=1).set_titles('')  # set_titles('{row_name} {col_name}')  # aspect = .65
                    if col_order == ['all']:
                        g.set(xlabel='')
                        # [ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor') for ax in g.axes.flat]
                    else:
                        [ax.set_xticks([.5]) for ax in g.axes.flat]
                        for axs_row in g.axes:
                            axs_row[0].set_xticklabels(['Optimistic'])
                            axs_row[1].set_xticklabels(['Pessimistic'])
                    [ax.set_ylabel(ylab) for ax in g.axes[:, 0]]
                    plt.savefig('figs/{}_{}_model_comparison_{}_{}.pdf'.format(protocol, '_'.join(col_order), ylab.replace('\n', ' '), figtit))

    if not plot_detail:
        return df_list, code_order, info, reduced_activities, all_dists, all_rda

    # the ones I've commented out are what I tried for plotting the Manip - No Manip differences on the same axes
    # [(rqfig, rqaxs), (refig, reaxs), (sfig, saxs), (spfig, spaxs), (mfig, maxs)] = [plt.subplots(
    #     3, n_dists * 3 + 1, figsize=(n_dists * 3 + 1.5, 3 * 1.1), sharey=True, sharex='col', gridspec_kw={'width_ratios': [2, 2, .75] * n_dists + [1.5]}) for _ in range(5)]
    # [(qfig, qaxs), (efig, eaxs)] = [plt.subplots(
    #     3, n_dists * 2 + 1, figsize=(n_dists * 1.7 + 1.5, 3 * 1.1), sharey=True, sharex='col', gridspec_kw={'width_ratios': [2, .75] * n_dists + [1.5]}) for _ in range(2)]
    [(rqfig, rqaxs), (refig, reaxs), (sfig, saxs), (spfig, spaxs), (mfig, maxs)] = [plt.subplots(
        3, n_dists * 3, figsize=(n_dists * 3.5, 3 * 1.1), sharey=True, sharex='col', gridspec_kw={'width_ratios': [2, 2, .75] * n_dists}) for _ in range(5)]
    [(qfig, qaxs), (efig, eaxs)] = [plt.subplots(
        3, n_dists * 2, figsize=(n_dists * 2, 3 * 1.1), sharey=True, sharex='col', gridspec_kw={'width_ratios': [2, .75] * n_dists}) for _ in range(2)]
    [(qlfig, qlaxs), (elfig, elaxs)] = [plt.subplots(n_dists, 6, figsize=(7.75, n_dists * 1.1), sharey='col',
                                                     gridspec_kw={'width_ratios': [2, 2, 2, 2, 2, .75]}) for _ in range(2)]

    # FROM LEARNING_RULE_FIG
    # set up distributional value predictors
    N_CHAN = half_preds * 2  #n_preds
    alpha_avg = 0.12
    rng = np.random.default_rng(seed=1)

    # get rewards efficiently
    n_trials = 2000
    bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    bounds = (bin_edges[0], bin_edges[-1])
    stretch = 1.05  # for x-axis

    opt_palette = {'Pessimistic': '#BA55D3', 'Optimistic': '#0047AB'}
    rgb = [mcol.to_rgb(x) for x in opt_palette.values()]
    uniform = mcol.LinearSegmentedColormap.from_list('', np.array(rgb), 256)
    ucols = get_colors(256, cmap=uniform)
    use_cmap = mcol.LinearSegmentedColormap.from_list('', ucols[np.array([0, 60, 105, 150, 255]), :], 256)
    colors = get_colors(N_CHAN, cmap=use_cmap)
    geno_colors = ['#0A704E'] * half_preds + ['#FFA001'] * half_preds

    static_args = {'n_trials': n_trials,
                   'alpha_avg': alpha_avg,
                   'N_CHAN': N_CHAN,
                   'taus': taus,  # taus
                   'colors': colors,
                   'bin_midpoints': bin_midpoints,
                   'bin_edges': bin_edges,
                   'bounds': bounds,
                   'grey': grey,
                   'stretch': stretch
                   }
    ### END LEARNING RULE FIG

    n_repeats = 100  # how many times to do the imputation

    # create dicts that will later become dfs
    template = {'Distribution': [],
                'Manipulation': [],
                'Tail': [],
                'Mean': [],
                'Variance': [],
                'Activity': []}

    quantile_data, reflected_data, expectile_data, reflected_expectile_data, sandra_data, sp_data, mikhael_data = \
        [deepcopy(template) for _ in range(7)]

    repreds = np.zeros((n_dists, half_preds * 2))
    rqpreds = np.zeros((n_dists, half_preds * 2))

    rexpectiles = np.stack([infer_dist(taus=taus, dist=distro) for distro in distros], axis=0)
    rexpectiles[:, :half_preds] = clamp_val - rexpectiles[:, :half_preds]

    rquantiles = np.stack([np.quantile(distro, taus) for distro in distros], axis=0)
    rquantiles[:, :half_preds] = clamp_val - rquantiles[:, :half_preds]

    for func in [np.mean, np.var]:
        for i_split in range(2):
            # plot example D1 and D2 predictors across trial types
            fig, axs = plt.subplots(1, 2, figsize=(4, 2))
            # first column quantiles, second column expectiles
            for ax, all_reflected, title in zip(axs, [rquantiles, rexpectiles], ['RQDRL', 'REDRL']):
                if i_split == 0:
                    ax.plot(np.arange(n_dists), func(all_reflected, axis=1), color='k')
                else:
                    for i_half, half in enumerate([np.arange(half_preds), np.arange(half_preds, half_preds * 2)]):
                        ax.plot(np.arange(n_dists), func(all_reflected[:, half], axis=1), color=np.array(geno_colors)[half][0])
                ax.set_xticks(np.arange(n_dists))
                ax.set_xticklabels(distro_names, rotation=45, ha='right', rotation_mode='anchor')
                ax.set_ylabel(func.__name__)
                ax.set_title(title)
            hide_spines()
            fig.tight_layout()
            fig.savefig(f'figs/{protocol}_activity_predictions_d1_d2_vs_tt_{func.__name__}_{i_split}.pdf', bbox_inches='tight')

    for i_distro, (distro, distro_name) in enumerate(zip(distros, distro_names)):

        quantiles = np.quantile(distro, taus)
        expectiles = infer_dist(taus=taus, dist=distro)
        print(distro_name)
        print(expectiles)
        #     expectiles = stats.expectile(distro, taus)

        for datalist, stat in zip([[quantile_data, reflected_data], [expectile_data, reflected_expectile_data]],
                                  [quantiles, expectiles]):
            for data in datalist:
                data['Distribution'].extend([distro_name] * 2)
                data['Manipulation'].extend(['No Stimulation'] * 2)
                data['Tail'].extend(['Pessimistic', 'Optimistic'])
                if stat is quantiles:
                    # treat each evenly-spaced quantile as defining a dirac delta on that point
                    data['Mean'].extend([np.mean(stat), np.mean(stat)])
                    data['Variance'].extend([np.mean((quantiles - np.mean(stat)) ** 2)] * 2)
                else:
                    #                 imputed = np.concatenate([infer_dist(taus=taus, expectiles=expectiles) for _ in range(n_repeats)])
                    imputed, loss = run_decoding(expectiles, taus, minv=reward_values[0], maxv=reward_values[-1])
                    #                 data['Variance'].extend([np.var(imputed)] * 2)
                    # now treat each sample like a dirac
                    data['Mean'].extend([np.mean(imputed), np.mean(imputed)])
                    data['Variance'].extend([np.mean((imputed - np.mean(imputed)) ** 2)] * 2)

            datalist[0]['Activity'].extend([np.mean(stat[:half_preds]), np.mean(stat[half_preds:])])
            # print([np.mean(stat[:half_preds]), np.mean(stat[half_preds:])])
            reflected = stat.copy()
            reflected[:half_preds] = clamp_val - reflected[:half_preds]
            datalist[1]['Activity'].extend([np.mean(reflected[:half_preds]), np.mean(reflected[half_preds:])])
            # print([np.mean(reflected[:half_preds]), np.mean(reflected[half_preds:])])

        #     # This is superfluous in this case. Since there's no manipulation the mean is the same,
        #     # it's just the direct activity readout that will change
        #     mean_reflected, imputed_preds = get_mean_reflected(reflected_quantile, half_preds, clamp_val)

        i_ax = i_distro * 3

        for data, activity, fig, axs, plot_taus in zip(
                [sandra_data, sp_data, mikhael_data], [sandra_activity, sandra_activity, mikhael_activity],
                [sfig, spfig, mfig], [saxs, spaxs, maxs], [otaus, otaus, [0.5]]):  # revert to taus if I want to see what they look like extremized

            data['Distribution'].extend([distro_name] * 2)
            data['Manipulation'].extend(['No Stimulation'] * 2)
            data['Tail'].extend(['Pessimistic', 'Optimistic'])

            # if activity is redrl_activity:
            #     inferred_V = activity.copy()
            #     inferred_V[1] = clamp_val - inferred_V[1]
            # else:
            inferred_V = activity[0, i_distro] - activity[1, i_distro]
            # print(inferred_V)

            if activity is sandra_activity:
                imputed, loss = run_decoding(inferred_V, taus, minv=reward_values[0], maxv=reward_values[-1])
                data['Mean'].extend([np.mean(imputed), np.mean(imputed)])
                data['Variance'].extend([np.mean((imputed - np.mean(imputed)) ** 2)] * 2)
                data['Activity'].extend([np.mean(inferred_V[:n_preds // 2]), np.mean(inferred_V[n_preds // 2:])])
                # print('I think this is it')
                print([np.mean(inferred_V[:n_preds // 2]), np.mean(inferred_V[n_preds // 2:])])
            else:
                inferred_V = np.max([inferred_V, 0])  # mean can't be less than zero
                data['Mean'].extend([inferred_V, inferred_V])
                data['Activity'].extend(activity[:, i_distro])
                if activity is mikhael_activity:
                    data['Variance'].extend([activity[0, i_distro] + activity[1, i_distro]] * 2)
                else:
                    data['Variance'].extend([np.nan, np.nan])

            axs[0, i_ax].set_title(distro_name)
            for i_row in range(n_rows):

                face_alpha = .3 if i_row > 0 else 1
                axs[i_row, i_ax].scatter(plot_taus, activity[0, i_distro], marker='o', color='#ffa001', alpha=face_alpha, edgecolors='none', s=30)
                axs[i_row, i_ax].scatter(plot_taus, activity[1, i_distro], marker='o', color='#0a704e', alpha=face_alpha, edgecolors='none', s=30)
                axs[i_row, i_ax].set_xticks([.1, .5, .9])

                axs[i_row, i_ax + 1].scatter(plot_taus, inferred_V, marker='o', color='#80808080', edgecolors='none', s=30)
                axs[i_row, i_ax + 1].set_xticks([.1, .5, .9])

                axs[i_row, i_ax + 2].hist(distro, orientation="horizontal", bins=np.arange(-.5, clamp_val + 1),
                                          color=grey, density=True)
                axs[i_row, i_ax + 2].set_xlim(0, 1)
                axs[i_row, i_ax + 2].axhline(np.mean(inferred_V), c='k', ls='-', lw=3)

            axs[i_row, i_ax].set_xlabel(r'$\tau$')
            axs[i_row, i_ax + 1].set_xlabel(r'$\tau$')
            axs[i_row, i_ax + 2].set_xlabel('Probability')

            # print(activity.shape)
            # print(half_preds // 2)
            # print(half_preds)

            for manip_pop, manip_inds, color, ls, tail in zip(
                    [1, 0], [np.arange(0, half_preds // 2), np.arange(half_preds // 2, half_preds)], [pess_color, opt_color],
                    ['--', 'dotted'], ['Pessimistic', 'Optimistic']):
                for manip_val, row, marker, manip, ec in zip([-clamp_val, clamp_val], [1, 2], ['x', '^'],
                                                             ['Inhibition', 'Excitation'], [None, 'none']):

                    tail_manip_preds = activity[:, i_distro].copy()
                    if data is sp_data:
                        tail_manip_preds[manip_pop, manip_inds] += manip_val
                    else:
                        tail_manip_preds[manip_pop] += manip_val
                    tail_manip_preds[tail_manip_preds < 0] = 0

                    # axs[row, i_ax].scatter(plot_taus, tail_manip_preds[0], marker=marker, color=geno_colors[-1], alpha=.3, edgecolors=ec)
                    # axs[row, i_ax].scatter(plot_taus, tail_manip_preds[1], marker=marker, color=geno_colors[0], alpha=.3, edgecolors=ec)

                    # plot the manipulated population in the correct genotype color
                    axs[row, i_ax].scatter(plot_taus, tail_manip_preds[manip_pop], marker=marker, color=geno_colors[manip_pop - 1], edgecolors=ec, s=30)
                    # plot the unmanipulated population in grey
                    # axs[row, i_ax].scatter(plot_taus, tail_manip_preds[int(not manip_pop)], marker=marker, color=grey, alpha=.3, edgecolors=ec)
                    V_manip = tail_manip_preds[0] - tail_manip_preds[1]
                    # plot the difference in opto/pess color
                    axs[row, i_ax + 1].scatter(plot_taus, V_manip, marker=marker, color=color, edgecolors=ec, s=30)
                    axs[row, i_ax + 2].axhline(np.mean(V_manip), ls=ls, color=color, lw=3)

                    data['Distribution'].append(distro_name)
                    data['Manipulation'].append(manip)
                    data['Tail'].append(tail)

                    if activity is sandra_activity:
                        # print(V_manip.shape)
                        # print(manip_inds)
                        # print(np.mean(V_manip[manip_inds]))
                        data['Activity'].append(np.mean(V_manip[manip_inds]))
                        imputed0, loss = run_decoding(V_manip, plot_taus, minv=reward_values[0], maxv=reward_values[-1])
                        data['Mean'].append(np.mean(imputed0))
                        data['Variance'].append(np.mean((imputed0 - np.mean(imputed0)) ** 2))
                    else:
                        V_manip = np.max([V_manip, 0])
                        data['Mean'].append(V_manip)
                        # print(tail_manip_preds[manip_pop])
                        data['Activity'].append(tail_manip_preds[manip_pop])
                        if activity is mikhael_activity:
                            data['Variance'].append(tail_manip_preds[0] + tail_manip_preds[1])
                        else:
                            data['Variance'].append(np.nan)

        for lfig, (oaxs, raxs), laxs, stat, all_reflected, method, datalist, sim_activity, rpreds in zip(
                [qlfig, elfig], [(qaxs, rqaxs), (eaxs, reaxs)], [qlaxs, elaxs], [quantiles, expectiles], [rquantiles, rexpectiles],
                ['quant', 'expec'], [[quantile_data, reflected_data], [expectile_data, reflected_expectile_data]],
                [rqdrl_activity, redrl_activity], [rqpreds, repreds]):

            reflected_preds = stat.copy()
            reflected_preds[:half_preds] = clamp_val - reflected_preds[:half_preds]
            rpreds[i_distro, :] = reflected_preds

            for i_row in range(n_rows):
                face_alpha = .3 if i_row > 0 else 1
                raxs[i_row, i_distro * 3].scatter(taus, reflected_preds, marker='o', color=geno_colors, alpha=face_alpha, edgecolors='none', s=30)
                raxs[i_row, i_distro * 3].set_xticks([.1, .5, .9])
            raxs[i_row, i_distro * 3].set_xlabel(r'$\tau$')

            # for axs, axcolor, dat, offset, n_ax in zip([oaxs, raxs], [grey, geno_colors], [stat, reflected_preds], [0, 1], [2, 3]):
            for axs, n_ax, offset in zip([oaxs, raxs], [2, 3], [0, 1]):
                i_ax = i_distro * n_ax + offset
                axs[0, i_ax].set_title(distro_name)
                for i_row in range(n_rows):
                    face_alpha = .3 if i_row > 0 else 1
                    axs[i_row, i_ax].scatter(taus, stat, marker='o', color=grey, alpha=face_alpha, edgecolors='none', s=30)
                    axs[i_row, i_ax].set_xticks([.1, .5, .9])
                    # axs[i_row, i_ax + 1].scatter(taus, dat, marker='o', color=axcolor) #, alpha=.3)
                    # axs[i_row, i_ax + 1].set_xticks([.1, .5, .9])
                    axs[i_row, i_ax + 1].hist(distro, orientation="horizontal", bins=np.arange(-.5, clamp_val + 1),
                                              color=grey, density=True)
                    axs[i_row, i_ax + 1].set_xlim(0, 1)
                    axs[i_row, i_ax + 1].axhline(np.mean(stat), c='k', ls='-', lw=3)
                axs[i_row, i_ax].set_xlabel(r'$\tau$')
                axs[i_row, i_ax + 1].set_xlabel('Probability')

            laxs[i_distro, 1].scatter(taus, stat, s=30, marker='o', color=colors, edgecolors='none')
            laxs[i_distro, 1].set_xticks([.1, .5, .9])

            # print(method, distro_name)
            # print('Variance ')
            # print(np.var(reflected_preds))
            # print('Mean')
            # print(np.mean(reflected_preds))
            laxs[i_distro, 2].scatter(taus, reflected_preds, s=30, marker='o', color=geno_colors)
            laxs[i_distro, 2].set_xticks([.1, .5, .9])

            laxs[i_distro, 3].scatter(tau_star[0], sim_activity[0, i_distro], s=30, marker='o', color=geno_colors[-1], edgecolors='none')
            laxs[i_distro, 3].scatter(tau_star[1], sim_activity[1, i_distro], s=30, marker='o', color=geno_colors[0], edgecolors='none')
            laxs[i_distro, 3].set_xticks([.1, .5, .9])

            laxs[i_distro, 4].scatter(taus, reflected_preds - all_reflected.mean(0), s=30, marker='o', color=geno_colors)
            laxs[i_distro, 4].set_xticks([.1, .5, .9])

            laxs[i_distro, 5].hist(distro, orientation="horizontal", bins=np.arange(-.5, clamp_val + 1),
                                   color=grey, density=True)
            laxs[i_distro, 5].set_xlim(0, 1)
            laxs[i_distro, 5].axhline(np.mean(stat), c='k', ls='-', lw=3)
            laxs[i_distro, 5].axhline(np.mean(reflected_preds), c=geno_colors[0], ls='-', lw=3)
            laxs[i_distro, 5].axhline(np.mean(reflected_preds), c=geno_colors[-1], ls=(0, (1, 1)), lw=3)

            static_args['gs'] = lfig._gridspecs[0]
            plt.figure(lfig)
            _ = plot_v(N_CHAN, distro, method, i_distro, 0, **static_args)

            for manip_inds, color, geno_color, ls, tail in zip([np.arange(0, half_preds), np.arange(half_preds, half_preds * 2)],
                                                   [pess_color, opt_color], [geno_colors[0], geno_colors[-1]],
                                                   ['--', 'dotted'], ['Pessimistic', 'Optimistic']):
                for manip_val, row, marker, manip, ec in zip([0, clamp_val], [1, 2], ['x', '^'],
                                                         ['Inhibition', 'Excitation'], [None, 'none']):
                # for manip_val, row, marker, manip in zip([-clamp_val, clamp_val], [1, 2], ['x', '^'],
                #                                          ['Inhibition', 'Excitation']):

                    #             tail_manip = distro.copy()
                    #             tail_manip[manip_inds] = manip_val
                    tail_manip_preds = stat.copy()
                    tail_manip_preds[manip_inds] = np.ones(half_preds) * manip_val

                    reflected_manip_preds = reflected_preds.copy()
                    reflected_manip_preds[manip_inds] = np.ones(half_preds) * manip_val
                    mean_reflected, imputed_preds = get_mean_reflected(reflected_manip_preds, half_preds, clamp_val)

                    raxs[row, i_distro * 3].scatter(taus[manip_inds], [manip_val] * len(manip_inds), marker=marker, color=geno_color, edgecolors=ec, s=30)

                    # for axs, act, dat, manip_mean, n_ax, offset in zip(
                    #         [oaxs, raxs], [tail_manip_preds, tail_manip_preds], [tail_manip_preds, reflected_manip_preds],
                    #         [np.mean(tail_manip_preds), mean_reflected], [2, 3], [0, 1]):
                    for axs, dat, manip_mean, n_ax, offset in zip([oaxs, raxs], [tail_manip_preds, reflected_manip_preds],
                                                                  [np.mean(tail_manip_preds), mean_reflected], [2, 3], [0, 1]):
                        i_ax = i_distro * n_ax + offset
                        # axs[row, i_ax].scatter(taus, tail_manip_preds, marker=marker, color=color, alpha=.3)
                        # axs[row, i_ax].scatter(taus, reflected_manip_preds, marker=marker, color=color)
                        # axs[row, i_ax].scatter(taus[manip_inds], act[manip_inds], marker=marker, color=color)
                        axs[row, i_ax].scatter(taus[manip_inds], dat[manip_inds], marker=marker, color=color, edgecolors=ec, s=30)
                        # axs[row, i_ax + 1].axhline(np.mean(tail_manip_preds), ls=ls, color=color, lw=3, alpha=.3)
                        # axs[row, i_ax + 1].axhline(mean_reflected, ls=ls, color=color, lw=3)
                        axs[row, i_ax + 1].axhline(manip_mean, ls=ls, color=color, lw=3)

                    for data in datalist:
                        data['Distribution'].append(distro_name)
                        data['Manipulation'].append(manip)
                        data['Tail'].append(tail)

                    datalist[0]['Activity'].append(np.mean(tail_manip_preds[manip_inds]))
                    datalist[1]['Activity'].append(np.mean(reflected_manip_preds[manip_inds]))
                    # print(np.mean(reflected_manip_preds[manip_inds]))

                    if stat is quantiles:
                        # treat each evenly-spaced quantile as defining a dirac delta on that point
                        datalist[0]['Mean'].append(np.mean(tail_manip_preds))
                        datalist[1]['Mean'].append(mean_reflected)
                        datalist[0]['Variance'].append(np.mean((tail_manip_preds - np.mean(tail_manip_preds)) ** 2))
                        datalist[1]['Variance'].append(np.mean((imputed_preds - mean_reflected) ** 2))
                    else:
                        #                     imputed0 = np.concatenate([infer_dist(taus=taus, expectiles=tail_manip_preds) for _ in range(n_repeats)])
                        #                     imputed1 = np.concatenate([infer_dist(taus=taus, expectiles=imputed_preds) for _ in range(n_repeats)])
                        imputed0, loss = run_decoding(tail_manip_preds, taus, minv=reward_values[0],
                                                      maxv=reward_values[-1])
                        imputed1, loss = run_decoding(imputed_preds, taus, minv=reward_values[0],
                                                      maxv=reward_values[-1])
                        #                     datalist[0]['Variance'].append(np.var(imputed0))
                        #                     datalist[1]['Variance'].append(np.var(imputed1))
                        for i_imp, imp in enumerate([imputed0, imputed1]):
                            datalist[i_imp]['Mean'].append(np.mean(imp))
                            datalist[i_imp]['Variance'].append(np.mean((imp - np.mean(imp)) ** 2))

    vfig, vax = plt.subplots(figsize=(1.3, 2))
    redf = pd.DataFrame({'tau': np.repeat(['Pessimistic', 'Optimistic'], half_preds),  # n_preds // 2),
                         'var': np.var(repreds, axis=0)})
    sns.pointplot(redf, x='tau', y='var', hue='tau', order=['Optimistic', 'Pessimistic'],
                  palette={'Pessimistic': pess_color, 'Optimistic': opt_color}, errwidth=4, scale=1)  # height=2, aspect=.65,
    vax.legend().remove()
    hide_spines()
    vfig.savefig(f'figs/{protocol}_REDRL_var.pdf', bbox_inches='tight')


    if cat:

        # cfig, caxs = plt.subplots(3, n_dists + 1, figsize=(n_dists * 1.75 + 1.5, 3*1.6), sharey=True, gridspec_kw={'width_ratios': [1.75]*3 + [1.5]})
        # cumfig, cumaxs = plt.subplots(3, n_dists + 1, figsize=(n_dists * 1.75 + 1.5, 3*1.6), sharey=True, gridspec_kw={'width_ratios': [1.75]*3 + [1.5]})
        # lapfig, lapaxs = plt.subplots(3, n_dists + 1, figsize=(n_dists * 1.75 + 1.5, 3*1.6), sharey=True, gridspec_kw={'width_ratios': [1.75]*3 + [1.5]})
        cfig, caxs = plt.subplots(3, n_dists, figsize=(n_dists * 1.75, 3*1.6), sharey=True)
        cumfig, cumaxs = plt.subplots(3, n_dists, figsize=(n_dists * 1.75, 3*1.6), sharey=True)
        lapfig, lapaxs = plt.subplots(3, n_dists, figsize=(n_dists * 1.75, 3*1.6), sharey=True)

        categorical_data, cumulative_data, laplace_data = [deepcopy(template) for _ in range(3)]

        for i_ax, (hist, cumhist, distro_name) in enumerate(zip(histos, cum_histos, distro_names)):

            for i_row in range(n_rows):
                caxs[i_row, i_ax].bar(bin_centers, hist, ec=grey, fc=grey, alpha=.5)

                cumaxs[i_row, i_ax].plot(bin_centers, cumhist, c=grey, lw=2)
                cumaxs[i_row, i_ax].fill_between(bin_centers, cumhist, fc=grey, alpha=.5)

                lapaxs[i_row, i_ax].plot(bin_centers, 1 - cumhist, c=grey, lw=2)
                lapaxs[i_row, i_ax].fill_between(bin_centers, 1 - cumhist, fc=grey, alpha=.5)

            weighted_mean = np.average(bin_centers, weights=hist)

            #     cat_activity = np.repeat(np.vstack(histos), 2, axis=0) @ tuning_curves #* 10  # times just for scaling
            #     cum_activity = np.repeat(np.vstack(cum_histos), 2, axis=0) @ tuning_curves #* 10
            #     lap_activity = np.repeat(1 - np.vstack(cum_histos), 2, axis=0) @ tuning_curves #* 10
            cat_act = cat_activity[i_ax * n_dup, :]
            cum_act = cum_activity[i_ax * n_dup, :]
            categorical_data['Activity'].extend([np.mean(cat_act[manip_neuron]) for manip_neuron in manip_neurons])
            cumulative_data['Activity'].extend([np.mean(cum_act[manip_neuron]) for manip_neuron in manip_neurons])
            laplace_data['Activity'].extend([np.mean(1 - cum_act[manip_neuron]) for manip_neuron in manip_neurons])

            for fig, axs, stat, ylab, data in zip([cfig, cumfig, lapfig], [caxs, cumaxs, lapaxs],
                                                  [hist, cumhist, 1 - cumhist],
                                                  ['Probability', '"Cumulative\nProbability"', '"Exceedance\nProbability"'],
                                                  [categorical_data, cumulative_data, laplace_data]):

                axs[0, i_ax].set_title(distro_name)
                for i_row in range(n_rows):
                    #             axs[i_row, i_ax].bar(bin_centers, hist, ec=grey, fc=grey, alpha=.5)
                    #         axs[i_row, i_ax].hist(distro, bins=bins, color=grey, density=True, alpha=.5)
                    #         axs[i_row, i_ax].scatter(bin_centers, hist, marker='o', color=grey, alpha=.2)
                    axs[i_row, i_ax].axvline(weighted_mean, c='k', ls='-', lw=3)
                    axs[i_row, i_ax].set_xticks(reward_values)

                axs[i_row, i_ax].set_xlabel(r'Magnitude ($\mu$L)')

                data['Distribution'].extend([distro_name] * 2 * 3)
                data['Manipulation'].extend(np.repeat(['No Stimulation', 'Inhibition', 'Excitation'], 2))
                data['Tail'].extend(np.tile(['Pessimistic', 'Optimistic'], 3))
                data['Mean'].extend([weighted_mean] * 2)
                data['Variance'].extend([np.sum(hist * (bin_centers - weighted_mean) ** 2)] * 2)

                if i_ax == 0:
                    axs[1, 1].set_title('Inhibition')
                    axs[2, 1].set_title('Excitation')

                    for i_row in range(n_rows):
                        axs[i_row, 0].set_ylabel(ylab)

                    fig.subplots_adjust(hspace=.6)

            for manip_val, row, marker, manip in zip([floor, ceiling], [1, 2], ['x', '^'], ['Inhibition', 'Excitation']):
                for manip_bin, manip_neuron, color, ls, tail in zip(manip_bins, manip_neurons, [pess_color, opt_color],
                                                                    ['--', 'dotted'], ['Pessimistic', 'Optimistic']):
                    #             manip_hist = counts.copy().astype(np.float16)
                    manip_hist = hist.copy()
                    manip_hist[manip_bin] = manip_val
                    if np.sum(manip_hist) == 0:
                        manip_hist = np.ones(nb) / nb
                    else:
                        manip_hist = manip_hist / np.sum(manip_hist)  # renormalize probability distribution
                    manip_weighted_mean = np.average(bin_centers, weights=manip_hist)

                    # plot categorical code (histogram)
                    caxs[row, i_ax].bar(bin_centers, manip_hist, ec=color, fill=False, ls=ls, alpha=.5)
                    #             axs[row, i_ax].scatter(bin_centers, manip_hist, marker=marker, color=color)
                    caxs[row, i_ax].axvline(manip_weighted_mean, ls=ls, color=color, lw=3)

                    categorical_data['Mean'].append(manip_weighted_mean)
                    categorical_data['Variance'].append(np.sum(manip_hist * (bin_centers - manip_weighted_mean) ** 2))

                    cat_act = manip_hist @ tuning_curves
                    categorical_data['Activity'].append(np.mean(cat_act[manip_neuron]))

                    #           # compute cumulative histogram and laplace code (1 - cumulative probability)
                    manip_cumhist = cumhist.copy()
                    manip_cumhist[manip_bin] = manip_val
                    imputed_hist = np.diff(manip_cumhist, prepend=0)

                    cum_act = manip_cumhist @ tuning_curves
                    cumulative_data['Activity'].append(np.mean(cum_act[manip_neuron]))

                    manip_laphist = 1 - cumhist
                    manip_laphist[manip_bin] = manip_val
                    imputed_laphist = np.diff(1 - manip_laphist, prepend=0)

                    lap_act = manip_laphist @ tuning_curves
                    laplace_data['Activity'].append(np.mean(lap_act[manip_neuron]))

                    for axs, cum, imp, data in zip([cumaxs, lapaxs], [manip_cumhist, manip_laphist],
                                                   [imputed_hist, imputed_laphist], [cumulative_data, laplace_data]):

                        if np.any(imp < 0):
                            imp -= np.amin(imp)
                        if np.sum(imp) == 0:
                            manip_renorm = np.ones(nb) / nb
                        else:
                            manip_renorm = imp / np.sum(imp)
                        cum_manip_weighted_mean = np.average(bin_centers, weights=manip_renorm)

                        # alternatively, I could plot manip_renorm rather than cum, but I think this gives
                        # a better intuition behind what's going on
                        axs[row, i_ax].plot(bin_centers, cum, c=color, lw=2, ls=ls)
                        axs[row, i_ax].fill_between(bin_centers, cum, fc=color, alpha=.5)
                        axs[row, i_ax].axvline(cum_manip_weighted_mean, ls=ls, color=color, lw=3)

                        data['Mean'].append(cum_manip_weighted_mean)
                        data['Variance'].append(np.sum(manip_renorm * (bin_centers - cum_manip_weighted_mean) ** 2))

        #             categorical_data['Distribution'].append(distro_name)
        #             categorical_data['Manipulation'].append(manip)
        #             categorical_data['Tail'].append(tail)

        cat_df = pd.DataFrame(categorical_data)
        cum_df = pd.DataFrame(cumulative_data)
        lap_df = pd.DataFrame(laplace_data)

    quantile_df = pd.DataFrame(quantile_data)
    reflected_df = pd.DataFrame(reflected_data)
    expectile_df = pd.DataFrame(expectile_data)
    reflected_expectile_df = pd.DataFrame(reflected_expectile_data)
    sandra_df = pd.DataFrame(sandra_data)
    sp_df = pd.DataFrame(sp_data)
    mikhael_df = pd.DataFrame(mikhael_data)

    if cat:
        df_list = [reflected_expectile_df, expectile_df, quantile_df, reflected_df, sandra_df, sp_df, mikhael_df, cat_df, lap_df, cum_df]
        axs_list = [reaxs, eaxs, qaxs, rqaxs, saxs, spaxs, maxs, caxs, laxs, cumaxs]
    else:
        df_list = [reflected_expectile_df, expectile_df, quantile_df, reflected_df, sandra_df, sp_df, mikhael_df]
        axs_list = [reaxs, eaxs, qaxs, rqaxs, saxs, spaxs, maxs]

    rel_nostim_dfs = []
    for use_df, sup in zip(df_list, code_order):
        use_df['Model'] = sup
        df_rel_nostim = use_df.copy()
        nostim_value = np.repeat(df_rel_nostim.loc[df_rel_nostim['Manipulation'] == 'No Stimulation', 'Mean'], 3).values
        nostim_variance = np.repeat(df_rel_nostim.loc[df_rel_nostim['Manipulation'] == 'No Stimulation', 'Variance'], 3).values
        df_rel_nostim['Mean'] -= nostim_value
        df_rel_nostim['Variance'] -= nostim_variance
        df_rel_nostim = df_rel_nostim[df_rel_nostim['Manipulation'] != 'No Stimulation']
        df_rel_nostim['excitation'] =  df_rel_nostim['Manipulation']
        df_rel_nostim.loc[df_rel_nostim['excitation'] == 'Excitation', 'excitation'] = True
        df_rel_nostim.loc[df_rel_nostim['excitation'] == 'Inhibition', 'excitation'] = False
        rel_nostim_dfs.append(df_rel_nostim)

    plot_model_predictions(rel_nostim_dfs, code_order, opt_palette, protocol)

    # for rel_nostim_df, axs, code_name in zip(rel_nostim_dfs, axs_list, code_order):
    #     axs[0, -1].remove()
    #     for excit, row in zip([False, True], [1, 2]):
    #         ax = axs[row, -1]
    #         sns.lineplot(data=rel_nostim_df[rel_nostim_df['excitation'] == excit], x='Distribution', y='Mean',
    #             hue='Tail', hue_order=['Pessimistic', 'Optimistic'], palette=opt_palette, ax=ax, legend=False)
    #         ax.set_xlim([-0.3, 2.3])
    #         ax.set_ylabel('{} mean:\nStim $-$ No Stim'.format(code_name))
    #         ax.set_xlabel('')
    #         ax.axhline(0, ls='--', color='k')
    #         if code_name == code_order[-1]:
    #             ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    #         else:
    #             ax.set_xticklabels([])

    hide_spines()
    for fig, axs, fname in zip([sfig, spfig, qfig, rqfig, efig, refig, mfig], [saxs, spaxs, qaxs, rqaxs, eaxs, reaxs, maxs],
                               ['Distributed_AU', 'Partial_Distributed_AU', 'Quantile', 'Reflected_Quantile',
                                'Expectile', 'Reflected_Expectile', 'AU']):

        axs[1, 2].set_title('Inhibition')
        axs[2, 2].set_title('Excitation')
        # axs[2, 3].legend()

        for i_row in range(n_rows):
            axs[i_row, 0].set_ylabel('Mean')

        # fig.subplots_adjust(hspace=.6)
        # hide_spines()
        # plt.tight_layout()
        fig.savefig(f'figs/{protocol}_{fname}_model.svg', bbox_inches='tight')
        fig.savefig(f'figs/{protocol}_{fname}_model.pdf', bbox_inches='tight')

    for fig, fname in zip([qlfig, elfig], ['quantile_learning', 'expectile_learning']):
        # fig.subplots_adjust(hspace=.6)
        # hide_spines()
        # plt.tight_layout()
        fig.savefig(f'figs/{protocol}_{fname}_model.svg', bbox_inches='tight')
        fig.savefig(f'figs/{protocol}_{fname}_model.pdf', bbox_inches='tight')

    # hide_spines()
    for fig, name in zip([cfig, cumfig, lapfig], ['categorical', 'cumulative', 'laplace']):
        for ax in fig.axes:
            ax.set_xlim([-1, 9])
        fig.tight_layout()
        fig.savefig(f'figs/{protocol}_{name}_model.svg', bbox_inches='tight')
        fig.savefig(f'figs/{protocol}_{name}_model.pdf', bbox_inches='tight')

    plt.show()


    return df_list, code_order, info, reduced_activities, all_dists, all_rda


def plot_model_predictions(rel_nostim_dfs, code_order, palette, protocol):

    row_order = ['Inhibition', 'Excitation']
    model_rel_nostim_df = pd.concat(rel_nostim_dfs)
    for dv_to_test in ['Mean', 'Variance']:
        g = sns.FacetGrid(data=model_rel_nostim_df, col='Model', col_order=code_order, row='Manipulation',
                          row_order=row_order, hue='Tail', hue_order=['Pessimistic', 'Optimistic'],
                          palette=palette, height=2, aspect=1)
        g.map_dataframe(sns.lineplot, x='Distribution', y=dv_to_test)
        g.set(xlim=(-0.3, 2.3))
        for i_row, row in enumerate(row_order):
            for i_col, col in enumerate(code_order):
                ax = g.axes[i_row, i_col]
                ax.set_title(col) #  if i_row == 0 else ax.set_title('')
                if i_col == 0:
                    # ax.set_ylabel('{} {}:\nStim $-$ No Stim'.format(row, dv_to_test.lower()))
                    ax.set_ylabel('{}:\nStim $-$ No Stim'.format(dv_to_test))
                if i_row == len(row_order) - 1:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                    ax.set_xlabel('')
                ax.axhline(0, ls='--', color='k')
        # plt.legend(loc=(1.04, 0))
        plt.savefig('figs/{}_model_comparison_stim_{}.pdf'.format(protocol, dv_to_test), bbox_inches='tight')

