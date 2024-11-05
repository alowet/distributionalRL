import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cmocean
import seaborn as sns
from scipy import stats
import pandas as pd
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm, AnovaRM
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper
from numpy.linalg import LinAlgError

# import scikit_posthocs as sp
import pingouin as pg
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.base import clone
from cycler import cycler
import tqdm

from streams import DataStream
from analysisUtils import consolidate_decode, simultaneous_decode, disjoint_decode, validate_ccgp, angle_between,\
    prepare_pseudomat, disjoint_regress
import warnings
import time
import multiprocessing as mp
import os
import sys
import copy
import itertools
import concurrent.futures

# sys.path.append('../utils')
sys.path.append('../utils')
from db import get_db_info
from plotting import add_cbar_and_vlines, set_share_axes, hide_spines, add_cbar, plot_confusion, plot_box, plot_stars, get_stars
from protocols import load_params
from paths import raise_print
from matio import loadmat
sys.path.append('../behavior_analysis')
from traceUtils import setUpLickingTrace


def compute_sig_frac_by_class(class_name, class_labels, corrs, neuron_info, corr_keys, n_psth_bins, inds=None):
    """
    Compute fraction of significant cells in each time bin, grouped by "class"
    In general, class_name will be "genotype", "lesion", or "subr" and class_labels
    will be e.g. ['D1-Cre', 'A2a-Cre', 'all'] or ['matched', 'lesioned']
    """

    if inds is None:
        inds = np.ones(len(neuron_info), dtype=bool)

    for key in corr_keys:
        corrs[key][class_name] = {}

        for to_ord in ['scram', 'ord']:
            corrs[key][class_name][to_ord] = {}

            # compute fraction in each class (e.g. lesioned vs. control or D1 vs D2)
            for i_class, class_label in enumerate(class_labels):

                this_ord = corrs[key][to_ord]['all']
                corrs[key][class_name][to_ord][class_label] = {}
                this_class = corrs[key][class_name][to_ord][class_label]
                this_class = setup_sig_frac(this_class, class_name, class_label, neuron_info, inds, this_ord, n_psth_bins)

                # compute fraction in each mouse
                for mouse_name in np.unique(neuron_info['names']):

                    corrs[key][class_name][to_ord][class_label][mouse_name] = {}
                    this_class = corrs[key][class_name][to_ord][class_label][mouse_name]
                    this_class = setup_sig_frac(this_class, class_name, class_label, neuron_info, inds, this_ord,
                                                n_psth_bins, sub_name='names', sub_label=mouse_name)

                    # compute fraction in each session
                    for fig_path in np.unique(neuron_info.loc[neuron_info['names'] == mouse_name, 'fig_paths']):

                        corrs[key][class_name][to_ord][class_label][mouse_name][fig_path] = {}
                        this_class = corrs[key][class_name][to_ord][class_label][mouse_name][fig_path]
                        this_class = setup_sig_frac(this_class, class_name, class_label, neuron_info, inds, this_ord,
                                                    n_psth_bins, sub_name='fig_paths', sub_label=fig_path)

    return corrs


def setup_sig_frac(this_class, class_name, class_label, neuron_info, inds, this_ord, n_psth_bins, sub_name=None, sub_label=None):
    # corrs[key][class_name][to_ord][class_label][mouse_name][fig_path] = {}
    # this_class = corrs[key][class_name][to_ord][class_label][mouse_name][fig_path]

    if sub_name is None: class_inds = inds
    else: class_inds = np.logical_and(neuron_info[sub_name] == sub_label, inds)
    if 'all' not in class_label.lower(): class_inds = np.logical_and(neuron_info[class_name] == class_label, class_inds)

    return compute_sig_frac(this_class, class_inds, this_ord, n_psth_bins)

def compute_sig_frac(this_class, class_inds, this_ord, n_psth_bins):

    this_class['inds'] = class_inds
    this_class['n'] = np.sum(class_inds)

    sig_correls = this_ord.sig[class_inds]
    pos_correls = this_ord.dat[class_inds, :, 2] > 0
    neg_correls = this_ord.dat[class_inds, :, 2] < 0
    pos_sig_correls = np.logical_and(sig_correls, pos_correls)
    neg_sig_correls = np.logical_and(sig_correls, neg_correls)

    for i_sign, (sign_correls, sign, n_sign_cells) in enumerate(zip(
            [sig_correls, pos_sig_correls, neg_sig_correls],
            ['All', 'Positive', 'Negative'],
            [np.tile(np.sum(class_inds), n_psth_bins), np.sum(pos_correls, axis=0),
             np.sum(neg_correls, axis=0)])):
        # if key in prerew_keys or to_ord == 'ord':
        # only plot once for postrew_keys, where scrambled control is meaningless
        sig_fraction = np.mean(sign_correls, axis=0)
        sig_freq = np.sum(sign_correls, axis=0)
        this_class[sign] = sig_fraction
        this_class[sign + '_freq'] = sig_freq
        this_class[sign + '_n'] = n_sign_cells
        this_class[sign + '_inds'] = sign_correls

    return this_class


def get_start_end_inds(use_corrs, corrs, corrs_seconds, psth_bin_centers, start_s, end_s, ind):

    if use_corrs == corrs:
        bin_width = np.mean(np.diff(psth_bin_centers))
        start_ind = np.argmin(np.abs(psth_bin_centers - bin_width / 2 - start_s))
        end_ind = np.argmin(np.abs(psth_bin_centers - bin_width / 2 - end_s))
    elif use_corrs == corrs_seconds:
        start_ind = ind
        end_ind = ind + 1

    return start_ind, end_ind


def plot_sig_frac_mouse_ttest(use_corrs, use_keys, class_name, class_labels, sign_labels,
                              start_ind, end_ind, mouse_colors, sign_colors, protocol, min_size=10):
    frac_dict = {'names': [],
                 'fracs': [],
                 'keys': [],
                 'sign': [],
                 class_name: []}

    for key in use_keys:
        for class_label in class_labels:
            for sign in sign_labels:
                incl_mouse_names = [mouse_name for mouse_name in mouse_colors.keys() if mouse_name in use_corrs[
                    key]['str_regions']['ord']['All Subregions'].keys() and use_corrs[key]['str_regions']['ord'][
                                        'All Subregions'][mouse_name][sign + '_n'][0] > min_size]
                n_mice = len(incl_mouse_names)
                class_fracs = np.array([use_corrs[key]['str_regions']['ord']['All Subregions'][mouse_name][sign]
                                        for mouse_name in incl_mouse_names]) * 100  # (n_mice, n_time_bins)
                shuff_fracs = np.array([use_corrs[key]['str_regions']['scram']['All Subregions'][mouse_name][sign]
                                        for mouse_name in incl_mouse_names]) * 100  # (n_mice, n_time_bins)
                per_class_fracs = np.mean(class_fracs[:, start_ind:end_ind], axis=1)
                per_shuff_fracs = np.mean(shuff_fracs[:, start_ind:end_ind], axis=1)
                frac_dict['names'].extend(incl_mouse_names)
                frac_dict['fracs'].extend(per_class_fracs - per_shuff_fracs)
                frac_dict['keys'].extend([key] * n_mice)
                frac_dict['sign'].extend([sign] * n_mice)
                frac_dict[class_name].extend([class_label] * n_mice)

    frac_df = pd.DataFrame(frac_dict)
    grid_kwargs = dict(col='keys', col_order=use_keys, aspect=.65, height=2, sharex=False,
                       gridspec_kws={'wspace': 0.1, 'hspace': 0.5})
    hue_kwargs = dict(x='sign', y='fracs', hue='names', palette=mouse_colors, zorder=0, size=8)
    mean_kwargs = dict(x='sign', y='fracs', hue='sign', palette=sign_colors, errwidth=4, errorbar=('ci', 95), scale=1.5,
                       dodge=.4)

    g = sns.FacetGrid(data=frac_df, **grid_kwargs)
    g.map_dataframe(sns.swarmplot, **hue_kwargs)  # style='dec_key', style_order=list(style_order),
    g.map_dataframe(sns.pointplot, **mean_kwargs).set_titles("")  # "{col_name}")
    # plt.legend(loc=(1.04, 0))

    for i_col, key in enumerate(use_keys):
        ax = g.axes[0, i_col]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
        ax.set_xlabel('')
        ax.set_title(key)
        ax.axhline(0, ls='--', c='#808080', zorder=-1)
        if i_col > 0:
            ax.spines['left'].set_color('none')
            ax.tick_params(axis='y', length=0)  # hide ticks without affecting leftmost axis
        else:
            ax.spines['left'].set_position(("axes", -0.15))
            ax.set_ylabel('Significant cells:\nData $-$ Shuffle (%)')

        data = frac_df[frac_df['keys'] == key]
        ps = []
        means = []
        sems = []
        for sign in sign_labels:
            #         wilc_stat, wilc_p = stats.wilcoxon(data.loc[data['sign'] == sign, 'fracs'])
            #         ps.append(wilc_p)
            sign_fracs = data.loc[data['sign'] == sign, 'fracs']
            stat, p = stats.ttest_1samp(sign_fracs, popmean=0)
            ps.append(p)
            means.append(np.mean(sign_fracs))
            sems.append(stats.sem(sign_fracs))
        print(ps)
        print(means)
        print(sems)
        plot_stars(ax, np.arange(len(sign_labels)), ps)

    hide_spines()
    [plt.savefig(os.path.join('..', 'neural-plots', '_'.join([
        protocol, 'mouse', '_'.join(use_keys), class_name, 'sig_fracs', 'sign', *sign_labels]) + fformat), bbox_inches='tight',
                 dpi=300)
     for fformat in ['.pdf', '.svg', '.png']]


def compute_sig_frac_sess_df(use_corrs, use_bin_centers, use_keys, class_name, class_labels, sign_labels, mice,
                             neuron_info, lt_start, lt_end, rew_start, rew_end,  min_size=40):

    time_dict = {'names': [],
                 'fig_paths': [],
                 'fracs': [],
                 'shuff': [],
                 'keys': [],
                 'sign': [],
                 'time': [],
                 'time_ind': [],
                 class_name: []}

    n_time = len(use_bin_centers)
    ords = ['ord', 'scram']
    n_ord = len(ords)

    for key in use_keys:
        for class_label in class_labels:
            for sign in sign_labels:
                for mouse_name in mice:
                    if mouse_name in use_corrs[key]['str_regions']['ord']['All Subregions'].keys():
                        tmp = use_corrs[key]['str_regions']['ord']['All Subregions'][mouse_name]
                        incl_fig_paths = [fig_path for fig_path in
                                          np.unique(neuron_info.loc[neuron_info['names'] == mouse_name, 'fig_paths'])
                                          if fig_path in tmp.keys() and tmp[fig_path][sign + '_n'][0] > min_size]
                        n_sess = len(incl_fig_paths)
                        for which_ord in ords:

                            fracs = np.array(
                                [use_corrs[key]['str_regions'][which_ord]['All Subregions'][mouse_name][fig_path][sign]
                                 for fig_path in incl_fig_paths]) * 100  # (n_mice, n_time_bins)
                            time_dict['fracs'].extend(fracs.flatten())

                        time_dict['names'].extend([mouse_name] * n_sess * n_time * n_ord)
                        time_dict['fig_paths'].extend(np.tile(np.repeat(incl_fig_paths, n_time), n_ord))
                        time_dict['shuff'].extend(np.repeat(ords, n_sess * n_time))
                        time_dict['keys'].extend([key] * n_sess * n_time * n_ord)
                        time_dict['sign'].extend([sign] * n_sess * n_time * n_ord)
                        time_dict['time'].extend(np.tile(use_bin_centers, n_sess * n_ord))
                        time_dict['time_ind'].extend(np.tile(np.arange(n_time), n_sess * n_ord))
                        time_dict[class_name].extend([class_label] * n_sess * n_time * n_ord)

    time_df = pd.DataFrame(time_dict)

    prerew_keys = ['mean', 'nolick_mean', 'resid_mean', 'var', 'resid_var']
    prerew_start_ind, prerew_end_ind = get_start_end_inds(use_corrs, use_corrs, None, use_bin_centers, lt_start, lt_end, None)
    postrew_start_ind, postrew_end_ind = get_start_end_inds(use_corrs, use_corrs, None, use_bin_centers, rew_start, rew_end, None)

    prerew_df = time_df[np.logical_and.reduce([
        np.isin(time_df['keys'], prerew_keys), time_df['time_ind'] >= prerew_start_ind, time_df['time_ind'] < prerew_end_ind])].groupby(
        ['names', 'fig_paths', 'shuff', 'keys', 'sign', class_name], as_index=False).mean().drop(columns=['time', 'time_ind'])
    postrew_df = time_df[np.logical_and.reduce([
        ~np.isin(time_df['keys'], prerew_keys), time_df['time_ind'] >= postrew_start_ind, time_df['time_ind'] < postrew_end_ind])].groupby(
        ['names', 'fig_paths', 'shuff', 'keys', 'sign', class_name], as_index=False).mean().drop(columns=['time', 'time_ind'])

    frac_df = pd.concat((prerew_df, postrew_df))
    diff_df = frac_df.pivot(index=['names', 'fig_paths', 'keys', 'sign', class_name], columns='shuff', values='fracs').reset_index()
    diff_df['fracs'] = diff_df['ord'] - diff_df['scram']
    diff_df = diff_df.drop(columns=['ord', 'scram'])

    mouse_time_df = time_df.groupby(['names', 'shuff', 'keys', 'sign', class_name, 'time', 'time_ind'], as_index=False).mean(numeric_only=True)
    mouse_frac_df = frac_df.groupby(['names', 'shuff', 'keys', 'sign', class_name], as_index=False).mean(numeric_only=True)
    mouse_diff_df = diff_df.groupby(['names', 'keys', 'sign', class_name], as_index=False).mean(numeric_only=True)

    return time_df, frac_df, diff_df, mouse_time_df, mouse_frac_df, mouse_diff_df


def plot_sig_frac_sess_lme(diff_df, use_keys, class_name, sign_labels, mouse_colors, sign_colors, protocol):

    plot_df = diff_df.groupby(['names', 'keys', 'sign', class_name], as_index=False).mean()
    plot_df = plot_df[np.isin(plot_df['sign'], sign_labels)]
    grid_kwargs = dict(col='keys', col_order=use_keys, aspect=.65, height=2, sharex=False,
                       gridspec_kws={'wspace': 0.1, 'hspace': 0.5})
    hue_kwargs = dict(x='sign', y='fracs', hue='names', palette=mouse_colors, zorder=0, size=8, order=sign_labels)
    mean_kwargs = dict(x='sign', y='fracs', hue='sign', palette=sign_colors, order=sign_labels, errwidth=4,
                       errorbar=('ci', 95), scale=1.5, dodge=.1)

    g = sns.FacetGrid(data=plot_df, **grid_kwargs)
    g.map_dataframe(sns.swarmplot, **hue_kwargs)  # style='dec_key', style_order=list(style_order),
    g.map_dataframe(sns.pointplot, **mean_kwargs).set_titles("")  # "{col_name}")
    # plt.legend(loc=(1.04, 0))

    for i_col, key in enumerate(use_keys):
        ax = g.axes[0, i_col]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
        ax.set_xlabel('')
        ax.set_title(key)
        ax.axhline(0, ls='--', c='#808080', zorder=-1)
        if i_col > 0:
            ax.spines['left'].set_color('none')
            ax.tick_params(axis='y', length=0)  # hide ticks without affecting leftmost axis
        else:
            ax.spines['left'].set_position(("axes", -0.15))
            ax.set_ylabel('Significant cells:\nData $-$ Shuffle (%)')

        ps = []
        for sign in sign_labels:
            data = diff_df[np.logical_and(diff_df['keys'] == key, diff_df['sign'] == sign)]
            model = mixedlm('fracs ~ 1', data, groups='names')
            mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
            print(mfit.summary())
            tmp = mfit.summary().tables[1]['P>|z|'][0]
            ps.append(1 if tmp == '' else float(tmp))
        plot_stars(ax, np.arange(len(sign_labels)), ps)
        print(ps)

    hide_spines()
    [plt.savefig(os.path.join('..', 'neural-plots', '_'.join([
        protocol, 'sess', '_'.join(use_keys), class_name, 'sig_fracs', 'sign', *sign_labels]) + fformat), bbox_inches='tight',
                 dpi=300)
     for fformat in ['.pdf', '.svg', '.png']]


def plot_conjunction_sig_frac_sess_by_class(neuron_info, corrs, protocol, mouse_colors=None, activity_type='spks'):
    # Compute test of independence for each session independently and compare to each session's expected number
    ind_dict = {'mouse': [], 'fig_path': [], 'actual': []}
    key = 'ord'
    sessions = np.unique(neuron_info['fig_paths'])
    n_sess = len(sessions)

    value_fracs = np.zeros(n_sess)
    rpe_fracs = np.zeros(n_sess)

    for i_sess, sess in enumerate(sessions):
        sess_subset = neuron_info['fig_paths'] == sess
        ind_dict['mouse'].append(neuron_info.loc[sess_subset, 'names'].iloc[0])
        ind_dict['fig_path'].append(sess)

        value_subset = np.logical_and(corrs['mean'][key]['all'].inds, sess_subset)
        rpe_subset = np.logical_and(corrs['rpe'][key]['all'].inds, sess_subset)
        combined_subset = np.logical_and(value_subset, rpe_subset)
        value_fracs[i_sess] = value_subset.sum() / sess_subset.sum()
        rpe_fracs[i_sess] = rpe_subset.sum() / sess_subset.sum()

        ind_dict['actual'].append(combined_subset.sum() * 100 / sess_subset.sum())
    ind_dict['expected'] = value_fracs * rpe_fracs * 100

    ind_df = pd.DataFrame(ind_dict)
    ind_df['diff'] = ind_df['actual'] - ind_df['expected']
    ind_df['dummy'] = 1
    model = mixedlm('diff ~ 1', ind_df, groups='mouse')
    mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
    print(mfit.summary())

    fig, ax = plt.subplots(figsize=(1.8, 2))
    sns.scatterplot(data=ind_df.groupby('mouse').mean(), x='expected', y='actual', hue='mouse', size='dummy',
                    palette=mouse_colors, ax=ax,
                    sizes=[50], legend=False)
    x = np.linspace(*ax.get_xlim(), 11)
    plt.plot(x, x, 'k--')
    plt.xlabel('Expected combined cells (%)')
    plt.ylabel('Actual combined cells (%)')
    hide_spines()

    plt.savefig(f'figs/{protocol}_{activity_type}_actual_vs_expected_by_session.pdf', bbox_inches='tight')



def plot_sig_frac_sess_by_class(class_name, class_labels, mouse_time_df, mouse_frac_df, diff_df, use_keys,
                                sign_labels, trace_dict, protocol, mouse_colors=None, activity_type='spks'):
    """
    :param class_name: e.g. 'helper' or 'lesion'
    :param class_labels: e.g. ['all'] or ['control', 'lesioned']
    :param mouse_time_df: pandas DataFrame with each session's significant fractions across time/key/sign/class/shuffling,
    averaged across sessions within mice
    :param mouse_frac_df: pandas DataFrame with each session's significant fractions across key/sign/class/shuffling,
    averaged across sessions within mice
    :param diff_df: pandas DataFrame with each session's difference in significant fractions between ordered and scrambled
     across key/sign/class, averaged across sessions within mice
    :param use_keys: e.g. ['mean', 'resid_var']
    :param sign_labels: e.g. ['All', 'Positive', 'Negative']
    :param trace_dict: for styling axes
    :param protocol: e.g. 'SameRewDist', for saving
    :param mouse_colors: dictionary mapping mouse names: colors
    :param activity_type: e.g. 'spks', for saving
    :return:
    """

    grid_kwargs = dict(height=2.2, row='keys', row_order=use_keys, col=class_name, col_order=class_labels, sharey=False)
    hue_kwargs = dict(y='fracs', hue='names', palette=mouse_colors, zorder=0, legend=False)
    mean_kwargs = dict(y='fracs', color='k', estimator='mean', errorbar=('ci', 95))

    for sign in sign_labels:
        data = mouse_time_df[np.logical_and(mouse_time_df['shuff'] == 'ord', mouse_time_df['sign'] == sign)]
        g1 = sns.FacetGrid(data=data, aspect=1.5, **grid_kwargs)
        g1.map_dataframe(sns.lineplot, x='time', **hue_kwargs)
        g1.map_dataframe(sns.lineplot, x='time', lw=3, err_kws={'alpha': .2}, **mean_kwargs).set_titles('')

        data2 = mouse_frac_df[mouse_frac_df['sign'] == sign]
        g2 = sns.FacetGrid(data=data2, aspect=1, **grid_kwargs)
        g2.map_dataframe(sns.lineplot, x='shuff', **hue_kwargs)
        g2.map_dataframe(sns.pointplot, x='shuff', linestyles='', **mean_kwargs).set_titles('')

        ylim, yticks = None, None
        for i_row, key in enumerate(use_keys):
            for i_col, class_label in enumerate(class_labels):
                for i, grid in enumerate([g1, g2]):

                    ax = grid.axes[i_row, i_col]

                    # on lineplot only
                    if i == 0:
                        scram_data = mouse_time_df[np.logical_and.reduce([
                            mouse_time_df['shuff'] == 'scram', mouse_time_df['keys'] == key,
                            mouse_time_df['sign'] == sign, mouse_time_df[class_name] == class_label])]

                        sns.lineplot(data=scram_data, x='time', ax=ax, lw=3, ls='--', err_kws={'alpha': .1}, **mean_kwargs)
                        ax.set_xticks([0, 2, 4])
                        ax.set_xticklabels([0, 2, 4])
                        setUpLickingTrace(trace_dict, ax=ax, override_ylims=True)
                        yticks = ax.get_yticks()
                        ylim = ax.get_ylim()

                    # on pointplot only
                    elif i == 1:

                        ax.set_yticks(yticks)
                        ax.set_ylim(ylim)

                        data3 = diff_df[np.logical_and.reduce([
                            diff_df['keys'] == key, diff_df['sign'] == sign, diff_df[class_name] == class_label])]
                        model = mixedlm('fracs ~ 1', data3, groups='names')
                        mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                        print(key, sign, class_label)
                        print(mfit.summary())
                        tmp = mfit.summary().tables[1]['P>|z|'][0]
                        pval = 1 if tmp == '' else float(tmp)
                        plot_stars(ax, [.5], [pval])

                    # on both plots
                    ax.spines['left'].set_position(("axes", -0.10 - i / 20))
                    ax.set_xlabel('')
                    if i_col == 0:
                        ax.set_ylabel(key)

        [grid.savefig(os.path.join('..', 'neural-plots', '_'.join([
            which, protocol, class_name, 'keys', *use_keys, 'sign', sign, 'sig_fracs', activity_type]) + fformat), bbox_inches='tight', dpi=300)
         for fformat in ['.pdf', '.png'] for which, grid in zip(['time', 'lt'], [g1, g2])]


def plot_sig_frac_mouse_by_class(class_name, class_labels, corrs, trace_dict, corr_keys, psth_bin_centers, protocol,
                                 class_colors=None, mouse_colors=None, min_size=40, activity_type='spks'):

    title_dict = {'mean': 'Mean', 'nolick_mean': 'Mean (Non-Licking Trials Only)', 'resid_mean': 'Residual Mean',
                  'var': 'Variance', 'resid_var': 'Residual Variance', 'rew': 'Reward', 'rpe': 'Reward Prediction Error',
                  'cvar': 'CVaR', 'resid_cvar': 'Residual CVaR'}

    sign_labels = ['Positive', 'Negative']
    # sign_colors = {'Negative': '#BA55D3', 'Positive': '#0047AB'}

    lt_start = 2
    lt_end = 3
    bin_width = np.mean(np.diff(psth_bin_centers))
    lt_start_ind = np.argmin(np.abs(psth_bin_centers - bin_width / 2 - lt_start))
    lt_end_ind = np.argmin(np.abs(psth_bin_centers - bin_width / 2 - lt_end))

    # print(lt_start_ind, lt_end_ind)
    corr_name = 'corrs' if lt_end_ind - lt_start_ind > 1 else 'corrs_seconds'

    prerew_keys = ['mean', 'nolick_mean', 'resid_mean', 'var', 'resid_var']

    for i_key, key in enumerate(corr_keys):

        frac_dict = {'names': [],
                     'fracs': [],
                     # 'keys': [],
                     'sign': [],
                     class_name: []}

        # summ_fig, summ_axs = plt.subplots(3, 1, figsize=(len(class_labels) * .8, 7), sharex=True,
        #                                   gridspec_kw={'hspace': .6})

        for i_sign, sign in enumerate(['All', 'Positive', 'Negative']):

            # fig, axs = plt.subplots(1, len(class_labels) * n_plots_per_class, figsize=(len(class_labels) * 5, 2),
            #                         sharey=True, gridspec_kw={'width_ratios': [2, 1] * len(class_labels)})
            all_class_frac_diffs = []
            incl_mouse_names_all = []
            # centers = []

            all_diff = pd.DataFrame()
            indiv_ps = np.ones(len(class_labels))
            for i_cl, class_label in enumerate(class_labels):
                incl_mouse_names = [mouse_name for mouse_name in mouse_colors.keys() if mouse_name in corrs[key][class_name]['ord'][class_label].keys() and
                                    corrs[key][class_name]['ord'][class_label][mouse_name][sign + '_n'][0] > min_size]
                n_mice = len(incl_mouse_names)

                incl_mouse_names_all.append(incl_mouse_names)
                class_fracs = np.array([corrs[key][class_name]['ord'][class_label][mouse_name][sign] for mouse_name in
                                        incl_mouse_names]) * 100  # (n_mice, n_time_bins)
                shuff_fracs = np.array([corrs[key][class_name]['scram'][class_label][mouse_name][sign] for mouse_name in
                                        incl_mouse_names]) * 100  # (n_mice, n_time_bins)


                if len(class_fracs) > 0:

                    if key in prerew_keys:
                        lt_class_fracs = np.mean(class_fracs[:, lt_start_ind:lt_end_ind], axis=1)
                        lt_shuff_fracs = np.mean(shuff_fracs[:, lt_start_ind:lt_end_ind], axis=1)
                    else:
                        lt_class_fracs = np.mean(class_fracs[:, lt_end_ind:], axis=1)
                        lt_shuff_fracs = np.mean(shuff_fracs[:, lt_end_ind:], axis=1)
                    # print(lt_class_fracs)
                    # print(lt_shuff_fracs)

                    df = pd.DataFrame(data={'lt_fracs': np.concatenate((lt_class_fracs, lt_shuff_fracs)),
                                            'mouse': incl_mouse_names * 2,
                                            'shuff': np.repeat([0, 1], len(incl_mouse_names)),
                                            class_name: [class_label] * len(incl_mouse_names) * 2})

                    frac_dict['names'].extend(incl_mouse_names)
                    frac_dict['fracs'].extend(lt_class_fracs - lt_shuff_fracs)
                    # frac_dict['keys'].extend([key] * n_mice)
                    frac_dict['sign'].extend([sign] * n_mice)
                    frac_dict[class_name].extend([class_label] * n_mice)

        frac_df = pd.DataFrame(frac_dict)
        trim_sign_labels = ['Positive', 'Negative']
        data = frac_df[np.isin(frac_df['sign'], trim_sign_labels)]
        # print(data.sort_values(by=[class_name, 'sign']))
        # g = sns.catplot(data, x='sign', y='fracs', hue='names', col=class_name, col_order=class_labels, order=trim_sign_labels,
        #                 palette=mouse_colors, height=2, aspect=.8, legend=False, sharey=False)
        # g.map_dataframe(sns.pointplot, x='sign', y='fracs', hue='sign', order=trim_sign_labels, palette=sign_colors).set_titles(
        #     '{col_name}')

        x = class_name
        x_labels = class_labels
        x_colors = class_colors
        col = 'sign'
        col_labels = trim_sign_labels

        g = sns.catplot(data, x=x, y='fracs', hue='names', col=col, col_order=col_labels, kind='swarm', size=8,
                        order=x_labels, palette=mouse_colors, height=3, aspect=.65, legend=False, sharey=False)
        g.map_dataframe(sns.pointplot, x=x, y='fracs', hue=x, order=x_labels, palette=x_colors, scale=1.5, dodge=0.4,
                        estimator='mean', errwidth=4, errorbar=('ci', 95)).set_titles('{col_name}')

        plt.suptitle(key)
        g.axes.flat[0].set_ylabel('Significant cells: Data $-$ Shuffle (%)')

        # for i_cls, class_label in enumerate(class_labels):
        for i, col_label in enumerate(col_labels):

            print(key, col_label, x_labels)
            ax = g.axes.flat[i]
            ax.axhline(0, ls='--', c='#808080')

            # diff_stat, diff_p = stats.ttest_ind(*[data.loc[np.logical_and(
            #     data[class_name] == class_label, data['sign'] == class_label), 'fracs'] for class_label in class_labels])
            # plot_stars(ax, [0.5], [diff_p])

            ps = []
            means = []
            sems = []
            # for sign_label in trim_sign_labels:
            for x_label in x_labels:
                sign_fracs = data.loc[np.logical_and(data[col] == col_label, data[x] == x_label), 'fracs']
                _, pvalue = stats.ttest_1samp(sign_fracs, 0)
                ps.append(pvalue)
                means.append(np.mean(sign_fracs))
                sems.append(stats.sem(sign_fracs))
            print(ps); print(means); print(sems)
            plot_stars(ax, np.arange(len(x_labels)), ps)

            #         ps = []
            #         for sign in sign_labels:
            #             wilc_stat, wilc_p = stats.ttest_1samp(data.loc[np.logical_and(
            #                 data[class_name] == class_label, data['sign'] == sign), 'fracs'], 0)
            #             ps.append(wilc_p)
            #         plot_stars(ax, np.arange(len(sign_labels)), ps)

            ax.set_xlabel('')
            ax.set_xticklabels(x_labels, rotation=45, ha='right', rotation_mode='anchor')
            ax.spines['left'].set_position(("axes", -0.15))

        hide_spines()
        [plt.savefig(os.path.join('..', 'neural-plots', '_'.join([
            protocol, key, class_name, 'sig_fracs', 'sign', *sign_labels, corr_name, activity_type]) + fformat), bbox_inches='tight', dpi=300)
         for fformat in ['.pdf', '.svg', '.png']]


def plot_stars_mouse_by_class(df, ax, rel_classes, class_name, depvar, groups, do_print=False):

    for i_rc, rel_class in enumerate(rel_classes):
        # print(all_diff[rel_class])
        # print(-1 in all_diff[rel_class].values)
        if class_name == 'str_regions' and -1 in df[rel_class].values:
            formula = '{} ~ C({})'.format(depvar, rel_class)
            model = mixedlm(formula, df, groups=groups)
            mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
            centers = np.unique(df[rel_class])
            # print(mfit.summary())
            pvals = mfit.summary().tables[1][-len(centers):-1]['P>|z|'].values
            pvals = [np.float64(x) if x != '' else 1. for x in pvals]
            centers = np.delete(centers, 0)  # class_labels.index(rel_class))
            if do_print:
                print(mfit.summary())
                print(pvals)
                print(centers)
            y = 1.15 + .15 * i_rc
            plot_stars(ax, centers, pvals, ytop_scale=y)
            ax.text(-1, y * ax.get_ylim()[1], rel_class)


def plot_shuff_bounds_by_class(class_name, class_labels, corrs, trace_dict, corr_keys, psth_bin_centers,
                               class_colors=sns.color_palette()):
    title_dict = {'mean': 'Mean', 'nolick_mean': 'Mean (Non-Licking Trials Only)', 'resid_mean': 'Residual Mean',
                  'var': 'Variance', 'resid_var': 'Residual Variance', 'rew': 'Reward',
                  'rpe': 'Reward Prediction Error'}
    alpha = .05
    n_psth_bins = len(psth_bin_centers)
    prerew_keys = ['mean', 'nolick_mean', 'resid_mean', 'var', 'resid_var']

    for i_class, class_label in enumerate(class_labels):
        for i_sign, sign in enumerate(['All']):
            fig, axs = plt.subplots(1, len(prerew_keys), figsize=(len(prerew_keys) * 3, 2.5))
            for i_key, key in enumerate(prerew_keys):

                ord_frac = corrs[key][class_name]['ord'][class_label][sign]
                ord_freq = corrs[key][class_name]['ord'][class_label][sign + '_freq']
                ord_n = corrs[key][class_name]['ord'][class_label][sign + '_n']

                # scram_frac = corrs[key][class_name]['scram'][class_label][sign]
                scram_freq = corrs[key][class_name]['scram'][class_label][sign + '_freq']
                scram_n = corrs[key][class_name]['scram'][class_label][sign + '_n']

                confint = proportion_confint(count=scram_freq, nobs=scram_n, alpha=alpha)
                out = np.array([proportions_ztest(count=[ord_freq[i_bin], scram_freq[i_bin]], nobs=[ord_n[i_bin], scram_n[i_bin]])
                                for i_bin in range(n_psth_bins)])
                reject = out[:, 1]

                ax = axs[i_key]
                setUpLickingTrace(trace_dict, ax=ax, override_ylims=True)
                ax.plot(psth_bin_centers, ord_frac * 100, color=class_colors[i_class])
                ax.fill_between(psth_bin_centers, confint[0] * 100, confint[1] * 100, alpha=0.2, color=class_colors[i_class])

                sigp = reject < (alpha / n_psth_bins)  # correct for multiple comparisons
                ax.scatter(psth_bin_centers[sigp], ax.get_ylim()[1] * np.ones(np.sum(sigp)), s=100, marker='v', color='g')

                if i_key == 0:
                    ax.set_ylabel('Fraction of\nsignificant cells (%)')
                else:
                    ax.set_ylabel('')
                if i_key == len(corr_keys) // 2:
                    ax.set_xlabel('Time from CS (s)')
                ax.set_title(title_dict[key])

            # fig.suptitle(class_label, y=1.05)
            hide_spines()

            plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
                [class_name, class_label, 'shuff_bounds']) + '.pdf'), bbox_inches='tight')
            plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
                [class_name, class_label, 'shuff_bounds']) + '.png'), bbox_inches='tight')



def plot_sig_frac_by_class(class_name, class_labels, corrs, trace_dict, corr_keys, psth_bin_centers, protocol, class_colors=None):
    title_dict = {'mean': 'Mean', 'nolick_mean': 'Mean (Non-Licking Trials Only)', 'resid_mean': 'Residual Mean',
                  'var': 'Variance', 'resid_var': 'Residual Variance', 'rew': 'Reward',
                  'rpe': 'Reward Prediction Error'}
    for i_sign, sign in enumerate(['All', 'Positive', 'Negative']):
        fig, axs = plt.subplots(1, len(corr_keys), figsize=(len(corr_keys) * 3, 2))
        for i_key, key in enumerate(corr_keys):
            ax = axs[i_key]
            setUpLickingTrace(trace_dict, ax=ax, override_ylims=True)
            if class_colors is not None:
                ax.set_prop_cycle(cycler(color=class_colors))
            class_fracs = np.array([corrs[key][class_name]['ord'][class_label][sign] for class_label in class_labels])
            ax.plot(psth_bin_centers, class_fracs.T * 100)
            ax.set_title(title_dict[key])
            if i_key == 0:
                ax.set_ylabel('Fraction of\nsignificant cells (%)')
            else:
                ax.set_ylabel('')
            # if i_key == len(corr_keys) - 1 and i_sign == 2:
            if i_key == len(corr_keys) // 2:
                ax.set_xlabel('Time from CS (s)')
        # if i_sign == 2:
        ax.legend(labels=class_labels, loc=(1.04, 0))

        hide_spines()
        plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
            [protocol, class_name, 'sig_fracs', sign]) + '.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
            [protocol, class_name, 'sig_fracs', sign]) + '.png'), bbox_inches='tight')

    # fig.tight_layout()


def simul_decode_by_class(class_name, class_labels, dec_dict, neuron_info, rois=['All Subregions'], min_size=40,
                          kern='linear', reg_C=5e-3, do_zscore=False, plot=False, shuffle=False):
    """
    :param class_name: string e.g. 'genotype', 'lesion', or 'helper'
    :param class_labels: list e.g. ['D1-Cre, 'A2a-Cre']
    :param dec_dict: dictionary of ccgp, pair, and cong
    :param neuron_info: DataFrame with class_name and fig_paths as columns
    :param rois: list e.g. ['lAcbSh', 'VLS', 'All Subregions']
    :param min_size: int with the minimum number of neurons per region to perform decoding
    :return: dec_dict, updated with pop_sizes and scores
    """

    fnum = 0

    _, fig_path_inds = np.unique(neuron_info['fig_paths'], return_index=True)
    unique_sessions = neuron_info.iloc[fig_path_inds]

    for time_bin in dec_dict.values():
        for i_grp, grp in enumerate(time_bin.values()):

            print(grp['name'])
            if shuffle:
                grp['scores']['shuff'] = {}
                score_dict = grp['scores']['shuff']
                # this is bad practice! shouldn't do this
                train_pers = [3]
                test_pers = [3]
            else:
                score_dict = grp['scores']
                train_pers = grp['train_pers']
                test_pers = grp['test_pers']

            for class_label in class_labels:

                print(class_label)
                if class_label not in score_dict:
                    score_dict[class_label] = {}

                if class_label == 'all':
                    class_inds = np.ones(neuron_info.shape[0], dtype=bool)
                else:
                    class_inds = neuron_info[class_name] == class_label

                for roi in rois:

                    if roi not in score_dict[class_label]:
                        score_dict[class_label][roi] = {}

                    if roi == 'All Subregions':
                        reg_inds = np.ones(neuron_info.shape[0], dtype=bool)
                    else:
                        reg_inds = neuron_info['str_regions'] == roi

                    mice = np.unique(neuron_info.loc[class_inds, 'names'])
                    for i_mouse, mouse_name in enumerate(mice):
                        if mouse_name not in score_dict[class_label][roi]:
                            score_dict[class_label][roi][mouse_name] = {}

                        mouse_inds = neuron_info['names'] == mouse_name
                        mouse_sessions = unique_sessions.loc[unique_sessions['names'] == mouse_name, 'fig_paths']

                        for fig_path in mouse_sessions.values:
                            sess_inds = neuron_info['fig_paths'] == fig_path
                            bool_inds = np.logical_and.reduce([class_inds, reg_inds, mouse_inds, sess_inds])

                            if np.sum(bool_inds) > min_size:

                                score_dict[class_label][roi][mouse_name][fig_path] = {}
                                use_dict = score_dict[class_label][roi][mouse_name][fig_path]
                                use_dict['pop_size'] = np.sum(bool_inds)

                                for i_per, (train_per, test_per) in enumerate(zip(train_pers, test_pers)):
                                    fnum += 1
                                    for i_key, key in enumerate(grp['keys']):
                                        if i_per == 0:
                                            use_dict[key] = {}
                                        per_key = '_'.join([str(train_per), str(test_per)])
                                        # print(use_dict)
                                        # print(key)
                                        # print(grp['resps'].keys())
                                        use_dict[key][per_key] = simultaneous_decode(
                                            grp['resps'][key][:, bool_inds], train_per, test_per, kern=kern,
                                            reg_C=reg_C, do_zscore=do_zscore, plot=plot, label=grp['name'], shuffle=shuffle)

                        # # breaks for debugging purposes
                        #                 break
                        #             break
                        #     break
                        # break

    return dec_dict


def pseudo_decode_by_class(class_name, class_labels, dec_dict, neuron_info, rois=['All Subregions'], n_sizes=2, by_mouse=True,
                           match_across_class=True, match_across_reg=False, min_size=40, n_splits=6, kern='linear', reg_C=5e-3, do_zscore=True):
    """
    :param class_name: string e.g. 'genotype', 'lesion', or 'mouse' (in which case by_mouse should be False)
    :param class_labels: list e.g. ['D1-Cre, 'A2a-Cre', 'all']
    :param dec_dict: dictionary of ccgp, pair, and cong
    :param neuron_info: DataFrame with class_name as a column
    :param rois: list e.g. ['lAcbSh', 'VLS', 'All Subregions']
    :param n_sizes: int of how many population sizes to use. Note that by default, this uses the same number of neurons
    for each class label (can be different for each roi)
    :param by_mouse: whether to further subdivide by mouse. Does not match number of neurons across mice
    :param match_across_class: bool. If True, match pop_sizes across class. If False, use max neurons possible
    :param match_across_reg: bool. If True, match pop_sizes across regions. If False, use max neurons possible
    :param min_size: only include a given mouse/mouse-region if there are more than this many neurons in it
    :param n_splits: number of splits of the data. 1 for test, n_splits -1 for training
    :param kern: kernel to use for svm
    :return: dec_dict, updated with pop_sizes and scores
    """

    all_sids = get_sids(neuron_info)
    # prune_class_labels = [class_label for class_label in class_labels if class_label != 'all']
    # prune rois is here because otherwise the pseudopopulation size gets massive and decoding takes a really long time
    # prune_rois = [roi for roi in rois if roi != 'All Subregions']

    # if match_across_class:
    #     assert len(prune_class_labels) > 0, "There must be at least one class to match across classes"

    max_sizes = pop_size_helper(class_name, class_labels, rois, neuron_info, by_mouse, match_across_class,
                                match_across_reg, min_size=min_size)
    print(max_sizes)

    for time_key, time_bin in dec_dict.items():

        if time_key == 'per' or n_sizes == 2:  # don't do bin pseudo decoding for testing effect of pop size

            for i_grp, grp in enumerate(time_bin.values()):

                # this was taking way to long to run for VLS (for bin_resps?), so try rbf instead
                # if grp['name'] == 'odor':
                #     use_kern = kern
                # else:
                #     use_kern = kern
                use_kern = kern

                print(grp['name'])
                for class_label in class_labels:

                    print(class_label)
                    if class_label not in grp['scores']:
                        grp['scores'][class_label] = {}

                    if class_label == 'all':
                        class_inds = np.ones(neuron_info.shape[0], dtype=bool)
                    else:
                        class_inds = neuron_info[class_name] == class_label
                    mice = np.unique(neuron_info.loc[class_inds, 'names'])

                    for roi in rois:

                        print(roi)

                        if roi not in grp['scores'][class_label]:
                            grp['scores'][class_label][roi] = {}

                        if roi == 'All Subregions':
                            reg_inds = np.ones(neuron_info.shape[0], dtype=bool)
                        else:
                            reg_inds = neuron_info['str_regions'] == roi

                        if by_mouse:
                            for i_mouse, mouse_name in enumerate(mice):

                                pop_sizes = np.logspace(np.log10(min_size), np.log10(max_sizes[class_label][roi][mouse_name]),
                                                        n_sizes).round().astype(np.int32)

                                if mouse_name not in grp['scores'][class_label][roi]:
                                    grp['scores'][class_label][roi][mouse_name] = {}
                                if 'pseudo' not in grp['scores'][class_label][roi][mouse_name]:
                                    grp['scores'][class_label][roi][mouse_name]['pseudo'] = {}

                                mouse_inds = neuron_info['names'] == mouse_name
                                use_dict = grp['scores'][class_label][roi][mouse_name]['pseudo']
                                bool_inds = np.logical_and.reduce([class_inds, reg_inds, mouse_inds])
                                # save_string = '_'.join([class_label, roi, mouse_name])

                                if np.sum(bool_inds) > min_size:
                                    # if (not match_across_class) and (not match_across_reg):
                                    #     pop_sizes = np.logspace(np.log10(min_size), np.log10(np.sum(bool_inds)), n_sizes).round().astype(np.int32)
                                    grp['scores'][class_label][roi][mouse_name]['pseudo'] = decode_by_class_helper(
                                        use_dict, all_sids, bool_inds, grp, pop_sizes, n_splits, kern=use_kern, reg_C=reg_C, do_zscore=do_zscore)
                        else:

                            pop_sizes = np.logspace(np.log10(min_size), np.log10(max_sizes[class_label][roi]['all_mice']),
                                                    n_sizes).round().astype(np.int32)

                            if 'all_mice' not in grp['scores'][class_label][roi]:
                                grp['scores'][class_label][roi]['all_mice'] = {}
                            use_dict = grp['scores'][class_label][roi]['all_mice']
                            bool_inds = np.logical_and(class_inds, reg_inds)
                            # save_string = '_'.join([class_label, roi])

                            if np.sum(bool_inds) > min_size:
                                grp['scores'][class_label][roi]['all_mice'] = decode_by_class_helper(
                                    use_dict, all_sids, bool_inds, grp, pop_sizes, n_splits, kern=use_kern, reg_C=reg_C, do_zscore=do_zscore)

    return dec_dict


def decode_by_class_helper(use_dict, all_sids, bool_inds, grp, pop_sizes, n_splits=6, kern='linear', reg_C=5e-3, do_zscore=True):

    for i_per, (train_per, test_per) in enumerate(zip(grp['train_pers'], grp['test_pers'])):
        # fnum += 1
        per_key = '_'.join([str(train_per), str(test_per)])

        # if grp['name'] == 'odor':
        #     use_dict['odor'] = {}
        #     use_dict['odor'][per_key] = disjoint_decode(
        #         grp['resps'][:, bool_inds], all_sids[bool_inds], pop_sizes=pop_sizes,
        #         n_splits=n_splits, train_per=train_per, test_per=test_per, kern=kern, reg_C=reg_C, do_zscore=do_zscore)
        # else:

        for i_key, key in enumerate(grp['keys']):
            if i_per == 0:
                use_dict[key] = {}

            # use_dict[key][per], _, _ = consolidate_decode(
            #     grp['resps'][key][:, bool_inds][..., per], fnum, pop_sizes=pop_sizes, color=grp['colors'][i_key],
            #     n_runs=50, replace=False)
            use_dict[key][per_key] = disjoint_decode(
                grp['resps'][key][:, bool_inds], all_sids[bool_inds], pop_sizes=pop_sizes,
                n_splits=n_splits, train_per=train_per, test_per=test_per, kern=kern, reg_C=reg_C, do_zscore=do_zscore)

        use_dict['pop_sizes'] = pop_sizes

    return use_dict


def subsample_trials(neuron_info, beh_df, beh_info, beh_resps, beh_dict, mask_shape, per_ind=3, name_key='names', date_key='file_dates'):

    nan_mask = np.zeros(mask_shape, dtype=bool)
    start_rewarded_tt = 2
    rng = np.random.default_rng(seed=1)

    for index, row in beh_df.iterrows():

        mouse_name = row['name']
        file_date = row['exp_date']
        fig_path = row['figure_path']
        neuron_inds = np.logical_and(neuron_info[name_key] == mouse_name, neuron_info[date_key].astype(np.int64) == file_date)

        beh_resp = beh_resps[:, beh_info['fig_paths'] == fig_path]
        beh_resp = beh_resp[:, 0, :, per_ind]
        #     print(beh_resp.shape)
        trials_per_rewarded_type = np.sum(~np.isnan(beh_resp[start_rewarded_tt:]), axis=1)

        correct = beh_dict['per']['mean']['scores']['all']['behavior'][mouse_name][fig_path]['Fixed vs. Variable'][
            '_'.join([str(per_ind)] * 2)][per_ind]
        # print(len(correct))
        # print(np.sum(trials_per_rewarded_type))
        assert (len(correct) == np.sum(trials_per_rewarded_type))

        n_incorrect = np.sum(~correct)
        correct_inds = np.flatnonzero(correct)
        retained_correct = rng.choice(correct_inds, size=n_incorrect, replace=False) if n_incorrect < np.sum(
            correct) else correct_inds
        retained_incorrect = np.flatnonzero(~correct)

        throwaway = np.ones(len(correct), dtype=bool)
        # throwaway[retained_incorrect] = False
        throwaway[np.concatenate((retained_correct, retained_incorrect))] = False

        reshape_throwaway = np.zeros((len(trials_per_rewarded_type), 1, mask_shape[2]), dtype=bool)
        start_tt = 0
        for i_tt, n_tt in enumerate(trials_per_rewarded_type):
            reshape_throwaway[i_tt, :, :n_tt] = throwaway[start_tt:start_tt + n_tt]
            start_tt += n_tt

        nan_mask[start_rewarded_tt:, neuron_inds, :] = reshape_throwaway

    return nan_mask


# def decode_by_class_helper(use_dict, all_sids, bool_inds, grp, save_string, fnum, pop_sizes):
def dec_dict_setup(cue_resps, all_spk_cnts, psth_bin_centers, protocol_info, kwargs, train_pers=[3], test_pers=[3],
                   nan_mask=np.zeros((6, 1, 90), dtype=bool), rewards=None):
    """
    :param cue_resps: array of shape (n_trace_types, n_neurons_or_facemap_predictors, max_n_trials_per_type, n_periods),
    containing either neuronal data or facemap data
    :param protocol_info: dictionary
    :param kwargs: dictionary
    :param train_pers: list of periods to index last dimension of cue_resps
    :param test_pers: list of periods to index last dimension of cue_resps
    :return:
    """
    n_trace_types = cue_resps.shape[0]
    all_spk_cnts = all_spk_cnts[:n_trace_types]

    dec_dict = {}

    for time_key, resps in zip(['per', 'bin'], [cue_resps, all_spk_cnts]):

        # CCGP setup
        ccgp_colors = ['#244b05', '#7cd411', '#4a9d05', '#2c6805']  # , '#ff7f3a', '#ffb380']
        # stack along axis 3 so that periods/bin is still the last dimension, and I can index as [..., per]
        ccgp_resps = {'Distribution CCGP 1': np.stack((resps[np.array([2, 4])], resps[np.array([3, 5])]), axis=3),
                      'Distribution CCGP 2': np.stack((resps[np.array([3, 5])], resps[np.array([2, 4])]), axis=3),
                      'Distribution CCGP 3': np.stack((resps[np.array([2, 5])], resps[np.array([3, 4])]), axis=3),
                      'Distribution CCGP 4': np.stack((resps[np.array([3, 4])], resps[np.array([2, 5])]), axis=3)
                      # 'Null CCGP 1': np.stack((resps[2:4], resps[4:6]), axis=3),
                      # 'Null CCGP 2': np.stack((resps[4:6], resps[2:4]), axis=3)
                      }

        # pairwise setup
        if protocol_info['protocol'] == 'SameRewDist':
            pair_keys = ['Fixed 1 vs. Variable 1', 'Fixed 2 vs. Variable 1', 'Fixed 1 vs. Variable 2',
                         'Fixed 2 vs. Variable 2', 'Fixed 1 vs. Fixed 2', 'Variable 1 vs. Variable 2']
            mean_keys = ['Nothing vs. Fixed', 'Nothing vs. Variable', 'Fixed vs. Variable']
        elif protocol_info['protocol'] == 'SameRewVar':
            pair_keys = ['Uniform 1 vs. Bimodal 1', 'Uniform 2 vs. Bimodal 1', 'Uniform 1 vs. Bimodal 2',
                         'Uniform 2 vs. Bimodal 2', 'Uniform 1 vs. Uniform 2', 'Bimodal 1 vs. Bimodal 2']
            mean_keys = ['Nothing vs. Uniform', 'Nothing vs. Bimodal', 'Uniform vs. Bimodal']
        else:
            raise Exception('Protocol not recognized')

        pair_order = [(2, 4), (3, 4), (2, 5), (3, 5), (2, 3), (4, 5)]
        pair_colors = ['#234d20', '#36802d', '#77ab59', '#c9df8a', '#ff7f3a', '#ffb380']
        pair_resps = {}

        for key, inds in zip(pair_keys, pair_order):
            pair_resps[key] = np.stack((resps[inds[0]], resps[inds[1]]), axis=0)
        # pair_pers = [late_trace_ind]
        # types_to_pair = np.array([2, 3, 4, 5])
        # for i, tt_i in enumerate(np.array(protocol_info['trace_type_names'], dtype='object')[types_to_pair]):
        #     for j, tt_j in enumerate(np.array(protocol_info['trace_type_names'], dtype='object')[types_to_pair][:i]):
        #         pair_resps[tt_j + ' v. ' + tt_i] = np.stack((resps[types_to_pair[i]], resps[types_to_pair[j]]), axis=0)

        # mean setup
        mean_colors = ['#e377c2', '#7f7f7f', '#17becf']
        mean_resps = {}
        mean_resps[mean_keys[0]] = np.concatenate((resps[np.array([0, 2])], resps[np.array([1, 3])]), axis=2)
        mean_resps[mean_keys[1]] = np.concatenate((resps[np.array([0, 4])], resps[np.array([1, 5])]), axis=2)
        mean_resps[mean_keys[2]] = np.concatenate((resps[np.array([2, 4])], resps[np.array([3, 5])]), axis=2)

        # congruency setup
        cong_keys = ['Congruent', 'Incongruent 1', 'Incongruent 2']
        cong_colors = ['#32cd32', '#ff7f3a', '#ffb380']
        cong_resps = {}

        # early_trace_ind = 2
        # cong_pers = [late_trace_ind]

        cong_resps[cong_keys[0]] = np.concatenate((resps[protocol_info['id_dist_inds'][0][1:]],
                                                   resps[protocol_info['id_dist_inds'][1][1:]]), axis=2)
        cong_resps[cong_keys[1]] = np.concatenate((resps[kwargs['id_mean_swap_inds'][0][0][1:]],
                                                   resps[kwargs['id_mean_swap_inds'][0][1][1:]]), axis=2)
        cong_resps[cong_keys[2]] = np.concatenate((resps[kwargs['id_mean_swap_inds'][1][0][1:]],
                                                   resps[kwargs['id_mean_swap_inds'][1][1][1:]]), axis=2)


        odor_keys = ['odor']  # protocol_info['trace_type_names']
        odor_colors = ['#6B0A34']
        # colors, _, _, _ = load_params(protocol_info['protocol'])
        # odor_colors = colors['colors'][:protocol_info['n_trace_types']]
        odor_resps = {'odor': resps}

        var_colors = ['#1f77b4', '#aec7e8']
        if rewards is not None:
            # Decode previous Variable trial reward size on the basis of all behavioral variables recorded on the next Variable trial.
            # add column of nans in front to shift behavior data back one trial
            tmp = np.concatenate((np.full((n_trace_types, resps.shape[1], 1, resps.shape[3]), np.nan), resps), axis=2)
            # add column of nans in back to match dims
            rewards_pad = np.concatenate((rewards, np.full((n_trace_types, rewards.shape[1], 1), np.nan)), axis=2)

            # get behavior responses following 2 uL reward
            tmp2 = tmp.copy()
            tmp2[rewards_pad != 2] = np.nan
            # get behavior responses following 6 uL reward
            tmp6 = tmp.copy()
            tmp6[rewards_pad != 6] = np.nan

            # indices corresponding to
            var_name = 'Bimodal' if protocol_info['protocol'] == 'SameRewVar' else 'Variable'
            var1_ind = protocol_info['trace_type_names'].index(f'{var_name} 1')
            var2_ind = protocol_info['trace_type_names'].index(f'{var_name} 2')

            var_resps = {'Variable 1': np.stack((tmp2[var1_ind], tmp6[var1_ind]), axis=0),
                         'Variable 2': np.stack((tmp2[var2_ind], tmp6[var2_ind]), axis=0)}
        else:
            var_resps = {}

        dec_dict[time_key] = {'ccgp': {}, 'pair': {}, 'cong': {}, 'mean': {}, 'odor': {}, 'var': {}}
        dist_colors = ['#74B72E', '#FF6600']  # for across vs. within distribution

        pk = ['Across distribution', 'Within distribution']

        for grp, keys, use_colors, resps, pooled_colors, pooled_keys in zip(dec_dict[time_key].keys(),
                                                               [ccgp_resps.keys(), pair_keys, cong_keys, mean_keys, odor_keys, var_resps.keys()],
                                                               [ccgp_colors, pair_colors, cong_colors, mean_colors, odor_colors, var_colors],
                                                               [ccgp_resps, pair_resps, cong_resps, mean_resps, odor_resps, var_resps],
                                                               [['#74B72E'], dist_colors, dist_colors, ['#C4AEAD', '#17BECF'], None, ['k']],
                                                               [['ccgp'], pk, pk, ['Across mean', 'Within mean'], ['odor'], ['var']]):
            dec_dict[time_key][grp] = {'name': grp,
                                     'keys': list(keys),
                                     'colors': use_colors,
                                     'resps': resps,
                                    # note that 'Fixed vs. Variable' is really a within-mean key, but keep it here for simplicity
                                     'within_dist_keys': ['Fixed 1 vs. Fixed 2', 'Variable 1 vs. Variable 2',
                                                          'Incongruent 1', 'Incongruent 2', 'Fixed vs. Variable',
                                                          'Uniform 1 vs. Uniform 2', 'Bimodal 1 vs. Bimodal 2', 'Uniform vs. Bimodal'],
                                     'pooled_colors': pooled_colors,
                                     'pooled_keys': pooled_keys,
                                     'scores': {}}

            if time_key == 'per':
                dec_dict[time_key][grp]['train_pers'] = train_pers  # [odor_ind, late_trace_ind],
                dec_dict[time_key][grp]['test_pers'] = test_pers
            else:
                dec_dict[time_key][grp]['train_pers'] = np.arange(all_spk_cnts.shape[3])
                dec_dict[time_key][grp]['test_pers'] = np.arange(all_spk_cnts.shape[3])
                dec_dict[time_key][grp]['time_bin_centers'] = psth_bin_centers

        if rewards is None:
            del dec_dict[time_key]['var']

    return dec_dict


def stim_gradient_setup(cue_resps, all_spk_cnts, psth_bin_centers, protocol_info, stim_locs, train_pers=[3], test_pers=[3]):
    """
    Special function, modeled off of dec_dict_setup, to use for decoding Fixed vs. Variable on stim on vs. stim off trials
    based on behavior
    :param cue_resps: array of shape (n_trace_types, n_neurons_or_facemap_predictors, max_n_trials_per_type, n_periods),
    containing either neuronal data or facemap data
    :param all_spk_cnts: array of shape (n_trace_types, n_neurons_or_facemap_predictors, max_n_trials_per_type, n_bins)
    :param psth_bin_centers: array of shape (n_bins), containing time bing centers
    :param protocol_info: dictionary
    :param kwargs: dictionary
    :param train_pers: list of periods to index last dimension of cue_resps
    :param test_pers: list of periods to index last dimension of cue_resps
    :return:
    """
    assert protocol_info['protocol'] == 'StimGradient'
    
    n_trace_types = protocol_info['n_trace_types']
    n_combo_types = cue_resps.shape[0]
    n_stim_locs = n_combo_types // n_trace_types
    assert n_stim_locs == len(stim_locs)

    dec_dict = {}

    for time_key, resps in zip(['per', 'bin'], [cue_resps, all_spk_cnts]):

        dec_dict[time_key] = {}

        # the only interesting thing is decoding Fixed from Variable across stimulation locatinos (nothing, ventral, dorsal)
        loc_resps = {}
        for i_stim_loc, stim_loc in enumerate(stim_locs):
            loc_resps[stim_loc] = resps[np.array([n_stim_locs + i_stim_loc, n_stim_locs * 2 + i_stim_loc])]

        dec_dict[time_key]['loc'] = {'name': 'loc',
                                 'keys': stim_locs,
                                 'colors': ['r', 'b', 'g'],
                                 'resps': loc_resps,
                                 'scores': {}}

        if time_key == 'per':
            dec_dict[time_key]['loc']['train_pers'] = train_pers  # [odor_ind, late_trace_ind],
            dec_dict[time_key]['loc']['test_pers'] = test_pers
        else:
            dec_dict[time_key]['loc']['train_pers'] = np.arange(all_spk_cnts.shape[3])
            dec_dict[time_key]['loc']['test_pers'] = np.arange(all_spk_cnts.shape[3])
            dec_dict[time_key]['loc']['time_bin_centers'] = psth_bin_centers

    return dec_dict


def pop_size_helper(class_name, prune_class_labels, prune_rois, neuron_info, by_mouse, match_across_class,
                    match_across_reg, min_size=40):

    mice = np.unique(neuron_info['names'])

    max_sizes = {class_label: {roi: {} for roi in prune_rois} for class_label in prune_class_labels}

    all_class_inds = [neuron_info[class_name] == class_label if class_label != 'all' else
                      np.ones(len(neuron_info), dtype=bool) for class_label in prune_class_labels]

    # all_mouse_inds = [neuron_info['names'] == mouse for mouse in mice]
    for use_cl in prune_class_labels:
        use_cl_inds = neuron_info[class_name] == use_cl if use_cl != 'all' else np.ones(len(neuron_info), dtype=bool)
        for use_roi in prune_rois:
            use_roi_inds = neuron_info['str_regions'] == use_roi if use_roi != 'All Subregions' else np.ones(
                len(neuron_info), dtype=bool)
            if by_mouse:
                for mouse in mice:
                    # print(prune_rois)
                    # print([(i for i in
                    #         [np.sum(np.logical_and.reduce([use_cl_inds,
                    #                                        neuron_info['str_regions'] == roi,
                    #                                        neuron_info['names'] == mouse])) for roi in prune_rois] if i > min_size)])
                    if match_across_class and not match_across_reg:
                        max_sizes[use_cl][use_roi][mouse] = min(i for i in
                            [np.sum(np.logical_and.reduce([neuron_info[class_name] == class_label, use_roi_inds,
                                                           neuron_info['names'] == mouse])) for class_label in prune_class_labels] if i > min_size)
                    elif match_across_reg and not match_across_class:
                        max_sizes[use_cl][use_roi][mouse] = min(i for i in
                            [np.sum(np.logical_and.reduce([use_cl_inds,
                                                           neuron_info['str_regions'] == roi,
                                                           neuron_info['names'] == mouse])) for roi in prune_rois] if i > min_size)
                    elif match_across_class and match_across_reg:
                        max_sizes[use_cl][use_roi][mouse] = min(i for i in
                            [np.sum(np.logical_and.reduce([neuron_info[class_name] == class_label,
                                                           neuron_info['str_regions'] == roi,
                                                           neuron_info['names'] == mouse])) for class_label in prune_class_labels for roi in prune_rois] if i > min_size)
                    else:
                        max_sizes[use_cl][use_roi][mouse] = np.sum(np.logical_and.reduce([use_cl_inds, use_roi_inds, neuron_info['names'] == mouse]))

            else:
                if match_across_class and not match_across_reg:
                    if use_cl != 'all':
                        max_sizes[use_cl][use_roi]['all_mice'] = np.amin(
                            [np.sum(np.logical_and(x, use_roi_inds)) for x in all_class_inds])
                    else:
                        max_sizes[use_cl][use_roi]['all_mice'] = np.sum(use_roi_inds)
                elif match_across_reg and not match_across_class:
                    max_sizes[use_cl][use_roi]['all_mice'] = np.amin(
                        [np.sum(np.logical_and(use_cl_inds, neuron_info['str_regions'] == roi)) for roi in prune_rois])
                elif match_across_class and match_across_reg:
                    max_sizes[use_cl][use_roi]['all_mice'] = np.amin(
                        [np.sum(np.logical_and(x, neuron_info['str_regions'] == roi)) for x in all_class_inds for roi in prune_rois])
                else:
                    max_sizes[use_cl][use_roi]['all_mice'] = np.sum(np.logical_and(use_cl_inds, use_roi_inds))

    return max_sizes


def make_dfs_helper(class_name, class_label, roi, use_dict, grp, pop_id, mouse_name=None, use_pop='max'):

    for i_key, key in enumerate(grp['keys']):
        for train_per, test_per in zip(grp['train_pers'], grp['test_pers']):
            per_key = '_'.join([str(train_per), str(test_per)])
            scores = use_dict[key][per_key][0]
            n_scores = scores.shape[1]
            n_sizes = scores.shape[0]
            if mouse_name is not None:
                prefix_dict = 'pseudo_mouse_dict'
                grp[prefix_dict]['mouse'].extend([mouse_name] * n_scores * n_sizes)
            else:
                prefix_dict = 'pseudo_dict'
            grp[prefix_dict][class_name].extend([class_label] * n_scores * n_sizes)
            grp[prefix_dict]['Subregion'].extend([roi] * n_scores * n_sizes)
            grp[prefix_dict]['grouping'].extend([key] * n_scores * n_sizes)
            grp[prefix_dict]['period'].extend([per_key] * n_scores * n_sizes)
            grp[prefix_dict]['pop_size'].extend(np.repeat(use_dict['pop_sizes'], n_scores))
            grp[prefix_dict]['Accuracy'].extend(scores.flatten())
            # max_pop = np.zeros(scores.shape, dtype=bool)
            # max_pop[-1, :] = 1
            this_pop = np.zeros(scores.shape, dtype=bool)
            if use_pop == 'max':
                this_pop[-1, :] = 1
            else:
                this_pop[use_dict['pop_sizes'] == use_pop, :] = 1
            grp[prefix_dict][use_pop].extend(this_pop.flatten())
            grp[prefix_dict]['pop_id'].extend(np.arange(pop_id, n_scores * n_sizes + pop_id))

    pop_id += n_scores * n_sizes
    return grp[prefix_dict], pop_id


def make_dfs_simul_helper(class_name, class_label, roi, use_dict, grp, pop_id, mouse_name, fig_path):

    prefix_dict = 'mouse_dict'
    nk = len(grp['keys'])
    np = len(grp['train_pers'])
    grp[prefix_dict]['mouse'].extend([mouse_name] * nk * np)
    grp[prefix_dict][class_name].extend([class_label] * nk * np)
    grp[prefix_dict]['Subregion'].extend([roi] * nk * np)
    grp[prefix_dict]['pop_size'].extend([use_dict['pop_size']] * nk * np)
    grp[prefix_dict]['fig_path'].extend([fig_path] * nk * np)
    grp[prefix_dict]['pop_id'].extend([pop_id] * nk * np)

    for i_key, key in enumerate(grp['keys']):
        grp[prefix_dict]['grouping'].extend([key] * np)
        for train_per, test_per in zip(grp['train_pers'], grp['test_pers']):
            per_key = '_'.join([str(train_per), str(test_per)])
            score = use_dict[key][per_key][0]
            grp[prefix_dict]['period'].append(per_key)
            grp[prefix_dict]['Accuracy'].append(score)

    pop_id += 1
    return grp[prefix_dict], pop_id


def make_dfs(class_name, class_labels, dec_dict, rois, by_mouse=True, pseudo=True, use_pop='max', stim=False):

    if by_mouse:
        prefix_dict = 'mouse_dict'
        prefix_dfs = 'mouse_dfs'
    else:
        prefix_dict = 'dict'
        prefix_dfs = 'dfs'

    if pseudo:
        prefix_dict = 'pseudo_' + prefix_dict
        prefix_dfs = 'pseudo_' + prefix_dfs

    for time_key, time_bin in dec_dict.items():
        if time_key == 'per' or pseudo == False:
            for grp in time_bin.values():

                # if not (pseudo and grp['name'] == 'odor'):
                print(grp['name'])

                # create pandas dataframe
                grp[prefix_dict] = {class_name: [],
                                       'grouping': [],
                                       'period': [],
                                       'pop_size': [],
                                       'Accuracy': [],
                                       'Subregion': [],
                                       'pop_id': []}
                # if stim:
                #     grp[prefix_dict]['excitation'] = []
                if by_mouse:
                    grp[prefix_dict]['mouse'] = []
                if pseudo:
                    grp[prefix_dict][use_pop] = []
                else:
                    grp[prefix_dict]['fig_path'] = []

                pop_id = 0
                for class_label in class_labels:
                    for roi in rois:
                        if by_mouse:
                            mice = [x for x in grp['scores'][class_label][roi].keys() if x != 'all_mice']
                            for mouse_name in mice:
                                if pseudo:
                                    use_dict = grp['scores'][class_label][roi][mouse_name]['pseudo']
                                    # print(use_dict.keys())
                                    if use_dict:
                                        grp[prefix_dict], pop_id = make_dfs_helper(class_name, class_label, roi, use_dict, grp,
                                                                                   pop_id, mouse_name, use_pop=use_pop)
                                else:
                                    for fig_path in grp['scores'][class_label][roi][mouse_name].keys():
                                        if fig_path != 'pseudo':
                                            use_dict = grp['scores'][class_label][roi][mouse_name][fig_path]
                                            if use_dict:  # if it isn't empty
                                                grp[prefix_dict], pop_id = make_dfs_simul_helper(
                                                    class_name, class_label, roi, use_dict, grp, pop_id, mouse_name, fig_path)
                        else:
                            use_dict = grp['scores'][class_label][roi]['all_mice']
                            if use_dict:
                                grp[prefix_dict], pop_id = make_dfs_helper(class_name, class_label, roi, use_dict, grp, pop_id, use_pop=use_pop)

                grp[prefix_dfs] = {}
                grp[prefix_dfs]['all'] = pd.DataFrame(grp[prefix_dict])
                # grp[prefix_dfs]['max_pop'] = {'disagg': {}, 'pool': {}}

                if not stim:
                    grp[prefix_dfs]['disagg'] = {}
                    grp[prefix_dfs]['pool'] = {}

                    for train_per, test_per in zip(grp['train_pers'], grp['test_pers']):  # take subset of df with maximum population size and correct period
                        per_key = '_'.join([str(train_per), str(test_per)])
                        if pseudo:
                            grp[prefix_dfs]['disagg'][per_key] = {}
                            grp[prefix_dfs]['pool'][per_key] = {}
                            grp[prefix_dfs]['disagg'][per_key][use_pop] = grp[prefix_dfs]['all'][
                                    (grp[prefix_dfs]['all']['period'] == per_key) & (grp[prefix_dfs]['all'][use_pop])]
                            # if use_pop == 'max':
                            #     grp[prefix_dfs]['disagg'][per_key][use_pop] = grp[prefix_dfs]['all'][
                            #         (grp[prefix_dfs]['all']['period'] == per_key) & (grp[prefix_dfs]['all']['max_pop'])]
                            # else:
                            #     grp[prefix_dfs]['disagg'][per_key][use_pop] = grp[prefix_dfs]['all'][
                            #         (grp[prefix_dfs]['all']['period'] == per_key) & (grp[prefix_dfs]['all']['pop_size'] == use_pop)]
                        else:
                            grp[prefix_dfs]['disagg'][per_key] = grp[prefix_dfs]['all'][grp[prefix_dfs]['all']['period'] == per_key]

                    # pool all the across-distribution decoding together and all the within-distribution decoding together
                    for per_key, disagg_df in zip(grp[prefix_dfs]['disagg'].keys(), grp[prefix_dfs]['disagg'].values()):
                        # new_df = 'pool_' + old_df
                        # if grp['name'] != 'odor':  # already included
                        grp[prefix_dfs]['pool'][per_key] = copy.deepcopy(disagg_df)

    return dec_dict


def run_stats(class_name, class_labels, rois, dec_dict, per_key, by_mouse=True, pseudo=True, use_pop='max'):
    """
    :param class_name: string e.g. 'genotype', 'lesion', or 'helper'.
    These act as categorical variables in the ANOVA
    :param class_labels: list of strings e.g. reg_labels, ['matched', 'lesion'], ['all']
    :param rois: list of strings with regions of interest
    :param dec_dict:  dictionary containing all the scores and dfs
    :per_key: str, train and test periods to use, joined by an underscore (e.g. '3_1')
    """

    if by_mouse:
        prefix_dfs = 'mouse_dfs'
    else:
        prefix_dfs = 'dfs'
    if pseudo:
        prefix_dfs = 'pseudo_' + prefix_dfs

    for grp in dec_dict['per'].values():

        # if grp['name'] != 'odor' or not pseudo:
        print(grp['name'])

        if 'stats' not in grp:
            grp['stats'] = {}

        if by_mouse:
            mice = np.unique(grp[prefix_dfs]['all']['mouse'])
            for mouse_name in mice:
                grp['stats'][mouse_name] = {roi: {'between_' + class_name: {}} for roi in rois}
                for class_label in class_labels:
                    grp['stats'][mouse_name][class_label] = {roi: {'disagg': {}, 'pool': {}} for roi in rois}
        else:
            # grp['stats']['all_mice'] = {class_label: {roi: {'disagg': {}, 'pool': {}} for roi in rois} for class_label in class_labels}
            grp['stats']['all_mice'] = {roi: {'between_' + class_name: {}, 'mouse_level': {}} for roi in rois}
            for class_label in class_labels:
                grp['stats']['all_mice'][class_label] = {roi: {'disagg': {}, 'pool': {}} for roi in rois}


        sub_df = grp[prefix_dfs]
        for df_id, use_keys in zip(['disagg', 'pool'], [grp['keys'], grp['pooled_keys']]):
            print(df_id)
            if grp['name'] not in ['odor', 'ccgp', 'var']:
                if by_mouse:
                    tmp_df = sub_df[df_id][per_key][use_pop]
                    for mouse_name in mice:
                        print(mouse_name)
                        use_df = tmp_df[tmp_df['mouse'] == mouse_name]
                        grp['stats'][mouse_name] = run_stats_helper(class_name, class_labels, rois, use_df, grp['stats'][mouse_name], df_id, use_keys)
                else:
                    print('all_mice')
                    use_df = sub_df[df_id][per_key][use_pop]
                    grp['stats']['all_mice'] = run_stats_helper(class_name, class_labels, rois, use_df, grp['stats']['all_mice'], df_id, use_keys)

    return dec_dict


# def report_lm(class_name, class_labels, rois, use_df, stat_grp, df_id, lme=False):
#     """
#     Run linear models on the decoding. This will be mixed ANOVAs (in the case of pseudopopulation data) or lienar mixed
#     effects models (in the case of mouse-level data)
#     Grouping is a repeated measure, since I set the random seed so each population is the same.
#     All other factors (genotype, lesion, etc.) are between-subjects measures, since those populations are different
#
#     :param class_name: e.g. 'genotype' or 'lesion'
#     :param class_labels: e.g. ['A2a-Cre', 'D1-Cre']
#     :param rois: e.g. reg_labels
#     :param use_df: e.g. grp['dfs']['max_pop']['disagg'][per]
#     :param lme: whether or not to use linear mixed effects model (should be True only for mouse-level analyses)
#     :return: prints ANOVA tables
#     """
#     for class_label in class_labels:
#         stat_grp = run_lm(class_name, class_label, 'Subregion', use_df, stat_grp, df_id, lme)
#
#     for roi in rois:
#         stat_grp = run_lm('Subregion', roi, class_name, use_df, stat_grp, df_id, lme)


def correlate_scores(dec_dict, beh_dict, per_keys, mouse_colors):
    for grp in dec_dict['per'].values():
        print(grp['name'])
        grp_data = {'mouse': [], 'fig_path': [], 'pop_size': [], 'key': [], 'period': [], 'score': [], 'beh_score': []}
        roi_data = grp['scores']['all']['All Subregions']
        beh_data = beh_dict['per'][grp['name']]['scores']
        paths = get_db_info()
        beh_fig_root = paths['behavior_fig_roots'][0]
        for mouse_name in roi_data.keys():
            for fig_path in roi_data[mouse_name].keys():
                if fig_path != 'pseudo':
                    fparts = fig_path.split(os.path.sep)
                    fd = fparts[-1]
                    beh_path = os.path.join(beh_fig_root, mouse_name, fd)
                    #                 print(beh_path)
                    #                 print(beh_data['control']['behavior'][mouse_name].keys())
                    for use_key in grp['keys']:
                        for per_key in roi_data[mouse_name][fig_path][use_key].keys():
                            grp_data['score'].append(roi_data[mouse_name][fig_path][use_key][per_key][0])
                            if mouse_name in beh_data['control']['behavior'].keys() and beh_path in \
                                    beh_data['control']['behavior'][mouse_name].keys():
                                grp_data['beh_score'].append(
                                    beh_data['control']['behavior'][mouse_name][beh_path][use_key][per_key][0])
                            elif mouse_name in beh_data['lesioned']['behavior'].keys() and beh_path in \
                                    beh_data['lesioned']['behavior'][mouse_name].keys():
                                grp_data['beh_score'].append(
                                    beh_data['lesioned']['behavior'][mouse_name][beh_path][use_key][per_key][0])
                            else:
                                grp_data['beh_score'].append(np.nan)
                        grp_data['key'].extend([use_key] * len(per_keys))
                        grp_data['period'].extend(per_keys)
                    grp_data['pop_size'].extend(
                        [roi_data[mouse_name][fig_path]['pop_size']] * len(grp['keys']) * len(per_keys))
                    grp_data['fig_path'].extend([fig_path] * len(grp['keys']) * len(per_keys))
                    grp_data['mouse'].extend([mouse_name] * len(grp['keys']) * len(per_keys))

        grp_df = add_grouping(grp_data, grp['within_dist_keys'])
        sess_avg_df = grp_df.groupby(['mouse', 'fig_path', 'pop_size', 'period', 'grouping'], as_index=False).mean()
        #     print(sess_avg_df[sess_avg_df['fig_path'] == sess_avg_df['fig_path'][0]])
        for grouping in np.unique(grp_df['grouping']):
            print(grouping)
            trim_df = sess_avg_df[sess_avg_df['grouping'] == grouping]
            for per_key in ['3_3']:
                per_df = trim_df[trim_df['period'] == per_key]
                plt.figure()
                sns.scatterplot(per_df, x='pop_size', y='score', hue='mouse', palette=mouse_colors, legend=False)
                model = mixedlm('score ~ pop_size', per_df, groups='mouse')
                mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                print(mfit.summary())
                plt.title(grp['name'] + grouping)

                plt.figure()
                sns.scatterplot(per_df, x='beh_score', y='score', hue='mouse', palette=mouse_colors, legend=False)
                model = mixedlm('score ~ beh_score', per_df.dropna(), groups='mouse')
                mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                print(mfit.summary())
                plt.title(grp['name'] + grouping)

        if len(np.unique(grp_df['grouping'])) == 2:
            print('diff')
            pivot_df = sess_avg_df.pivot(columns='grouping', index=['mouse', 'fig_path', 'pop_size', 'period'],
                                         values=['score', 'beh_score']).reset_index()
            #             print(pivot_df.keys())
            pivot_df.columns = [' '.join(col).strip() for col in pivot_df.columns.values]
            pivot_df['diff'] = pivot_df['score Across distribution'] - pivot_df['score Within distribution']
            pivot_df['beh_diff'] = pivot_df['beh_score Across distribution'] - pivot_df[
                'beh_score Within distribution']
            for per_key in ['3_3']:
                per_df = pivot_df[pivot_df['period'] == per_key]
                plt.figure()
                sns.scatterplot(per_df, x='pop_size', y='diff', hue='mouse', palette=mouse_colors, legend=False)
                model = mixedlm('diff ~ pop_size', per_df, groups='mouse')
                mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                print(mfit.summary())
                plt.title(grp['name'] + ' diff')

                plt.figure()
                sns.scatterplot(per_df, x='beh_diff', y='diff', hue='mouse', palette=mouse_colors, legend=False)
                model = mixedlm('diff ~ beh_diff', per_df.dropna(), groups='mouse')
                mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                print(mfit.summary())
                plt.title(grp['name'] + ' diff')

    hide_spines()

def run_lm(use_name, use_label, between, use_df, lme=False, test='lme', depvar='Accuracy', groups='mouse'):
    """
    Run linear models on the decoding accuracy. This will be two-way ANOVAs (in the case of pseudopopulation data) or linear mixed
    effects models/RM ANOVAs (in the case of mouse-level data)
    Grouping is a repeated measure, since I set the random seed so each population is the same.
    All other factors (genotype, lesion, etc.) are between-subjects measures, since those populations are different

    :param use_name: restrict model to this variable e.g. 'genotype' or 'lesion' or 'Subregion'
    :param use_label: restrict model to this label (for variable), e.g. 'A2a-Cre' or 'D1-Cre'
    :param between: between-subjects factor (e.g. 'Subregion' when use_name == 'lesion')
    :param use_df: e.g. grp['dfs']['max_pop']['disagg'][per]
    :param lme: whether or not to use mixed linear effects model (should be True only for mouse-level analyses)
    :return: fitted model
    """
    sub_df = use_df[use_df[use_name] == use_label]
    if len(sub_df) == 0:
        print('No data for {} = {} between {}. Skipping'.format(use_name, use_label, between))
        return

    if len(np.unique(sub_df[between])) > 1 and len(np.unique(sub_df['grouping'])) > 1:
        formula = f'{depvar} ~ C({between}) * C(grouping)'
    elif len(np.unique(sub_df[between])) > 1:
        formula = f'{depvar} ~ C({between})'
    elif len(np.unique(sub_df['grouping'])) > 1:
        formula = f'{depvar} ~ C(grouping)'
    else:
        formula = f'{depvar} ~ 1'
    print(formula)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if lme:
            if test == 'lme':
                # if use_name == 'genotype':
                #     # based off of the discussion here https://groups.google.com/g/pystatsmodels/c/wN3-vTd3Ld4, I believe this
                #     # correctly specifies a model in which mouse is nested within genotype
                #     # From the docs: https://www.statsmodels.org/stable/_modules/statsmodels/regression/mixed_linear_model.html
                #     # If the variance component is intended to produce random intercepts for disjoint subsets of a group,
                #     # specified by string labels or a categorical data value, always use '0 +' in the formula so that no
                #     # overall intercept is included.
                #     model = mixedlm(formula, sub_df, vc_formula={'mouse': '0 + C(mouse)'}, groups='genotype')
                # else:
                #     # otherwise, there is no nesting, and I should give each mouse its own random intercept
                #     # print(sub_df)
                model = mixedlm(formula, sub_df, groups=groups, re_formula='~grouping')
                try:
                    mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                except LinAlgError:
                    mfit = pd.DataFrame()
                except ValueError:
                    mfit = pd.DataFrame()
                # print(formula)
                # print(mfit.summary())
                # print(model.score(mfit.params_object))  # should be below 1e-2 to ensure convergence

            else:
                # print(sub_df)
                subj_means = sub_df.groupby([groups, 'grouping', between]).mean().reset_index()
                # print(subj_means)
                mfit = pg.mixed_anova(data=subj_means, dv=depvar, within='grouping', subject='pop_id', between=between,
                                       correction=True)

        else:
            model = ols(formula, sub_df).fit()
            mfit = anova_lm(model, typ=3, robust='hc3')

    return mfit


def run_stats_helper(class_name, class_labels, rois, use_df, stat_grp, df_id, keys):
    """
    For a single mouse or pseudopopulation, run stats on the disjoint (pseudo)-populations
    :param class_name:
    :param class_labels:
    :param rois:
    :param use_df:
    :param stat_grp:
    :param df_id:
    :return:
    """

    # within class_name/subregion, do mixed anova followed by posthoc tests
    for class_label in class_labels:
        # print('\n' + class_label)
        stat_grp[class_label]['between_roi'] = {}
        for roi in rois:
            if np.sum(use_df['Subregion'] == roi) > 0:
                subr_df = use_df[use_df['Subregion'] == roi]
                if len(subr_df) > len(keys):  # if n_train > 1. Otherwise, this comparison won't work
                    # print(keys)
                    if keys == ['odor']:
                        subr_df['Accuracy'] -= 1/6  # chance is 1 / n_odors
                        model = ols('Accuracy ~ 1', subr_df).fit()
                    elif keys == ['ccgp']:
                        subr_df['Accuracy'] -= .5 # chance is 1 / 2
                        model = ols('Accuracy ~ 1', subr_df).fit()
                    else:
                        model = ols('Accuracy ~ C(grouping)', subr_df).fit()
                    stat_grp[class_label][roi][df_id]['oneway'] = anova_lm(model, typ=3, robust='hc3')

        # ANOVA across all rois/groupings for each class separately
        if len(rois) > 1:
            try:
                stat_grp[class_label]['between_roi'][df_id] = run_lm(class_name, class_label, 'Subregion', use_df)
            except ValueError:  # must have at least one row in constraint matrix
                pass

    for roi in rois:
        # ANOVA across classes/groupings for each subregion separately
        # print(roi)
        # if len(class_labels) > 1:
        try:
            stat_grp[roi]['between_' + class_name][df_id] = run_lm('Subregion', roi, class_name, use_df)
        except ValueError:  # must have at least one row in constraint matrix
            pass

    return stat_grp


def plot_decode_by_class(class_name, class_labels, dec_dict, per_keys, rois=['All Subregions'], pseudo=True,
                         n_splits=6, rem=False, beh=False, use_pop='max', reg_C=5e-3, class_colors=None, activity_type='spks'):

    n_train = n_splits - 1

    if pseudo:
        prefix_df = 'pseudo_dfs'
    else:
        prefix_df = 'dfs'

    save_dir = get_save_dir(rem, beh)
    palette_options = sns.color_palette('husl', n_train * len(class_labels))
    # print(len(palette_options))

    # for time_key, time_bin in zip(dec_dict.keys(), dec_dict.values()):
        # for time_key, time_bin in zip(['bin'], [dec_dict['bin']]):
    for i_grp, grp in enumerate(dec_dict['per'].values()):

        hlines = 1 / grp['resps']['odor'].shape[0] if grp['name'] == 'odor' else .5

        for df_id, use_colors, use_keys in zip(['disagg', 'pool'], [grp['colors'], grp['pooled_colors']], [grp['keys'], grp['pooled_keys']]):

            cls_flag = True if grp['name'] == 'odor' or (grp['name'] == 'ccgp' and df_id == 'pool') else False

            row_name, row_order, col_name, col_order, hue, hue_order = get_row_col_hue(rois, class_name, class_labels, use_keys)
            # col_order = hue_order if len(row_order) == 1 and len(col_order) == 1 else col_order

            aspect = (len(use_keys) + 3) / 10 if len(use_keys) > 1 else .5
            grid_kwargs = dict(col=col_name, col_order=col_order, row=row_name, row_order=row_order, aspect=aspect,
                               height=2, sharex=False, gridspec_kws={'wspace': 0.1, 'hspace': 0.5})
            hue_kwargs = dict(x=hue, y='Accuracy', hue='pop_id', zorder=1, label=False)
            mean_kwargs = dict(x=hue, y='Accuracy', hue=hue, order=hue_order, hue_order=hue_order,
                               palette=use_colors, estimator='mean', errwidth=4, errorbar=('ci', 95))
            # print(hue, hue_order)
            if cls_flag: mean_kwargs['palette'] = class_colors

            for per_key in per_keys:
                print(grp['name'], df_id, use_keys, per_key)

                # in case of 'pool', aggregate within and across-distribution keys before plotting
                use_df = grp[prefix_df][df_id][per_key][use_pop]
                use_df = use_df[np.isin(use_df[class_name], class_labels)]

                # print(use_df.shape, use_df)
                agg_df = use_df.groupby(['Subregion', class_name, 'grouping', 'pop_id']).mean().sort_values(
                    by='grouping', key=lambda series: [use_keys.index(x) for x in series]).reset_index()
                agg_df['i_grouping'] = agg_df['grouping'].apply(lambda grping: use_keys.index(grping))
                agg_df['i_cls'] = agg_df[class_name].apply(lambda cls: class_labels.index(cls))
                # print(agg_df)
                # print(np.unique(agg_df['pop_id']))
                palette = {k: v for k, v in zip(np.unique(agg_df['pop_id']), palette_options)}
                # print(palette)

                g = sns.FacetGrid(data=agg_df, **grid_kwargs)
                g.map_dataframe(sns.pointplot, scale=1.5, dodge=0.4, **mean_kwargs).set_titles("")
                g.axes.flat[-1].legend(loc=(1.04, 0))
                if grp['name'] == 'odor' or (grp['name'] == 'ccgp' and df_id == 'pool'):
                    g.map_dataframe(sns.swarmplot, size=8, order=class_labels, palette=palette, **hue_kwargs)
                else:
                    g.map_dataframe(sns.lineplot, estimator=None, palette=palette, **hue_kwargs)

                g.set(xlim=(-.5, len(hue_order) - .5))
                g.fig.suptitle(grp['name'] + ' ' + per_key, y=1.2)
                arrange_grid(g, col_order, hue_order, hlines, None, None)

                # diff_df = agg_df[agg_df['grouping'] == use_keys[0]]
                # if grp['name'] not in ['odor', 'ccgp'] and pseudo and df_id == 'pool':
                #     diff_df['Accuracy'] -= agg_df.loc[agg_df['grouping'] == use_keys[1], 'Accuracy'].values

                int_df = agg_df.copy()
                int_df['Accuracy'] -= hlines

                for i_roi, roi in enumerate(rois):
                    print(roi)
                    roi_df = int_df[int_df['Subregion'] == roi]
                    intps = {k: [] for k in class_labels}

                    ax = g.axes[i_roi, 0]
                    ax.text(x=-2, y=ax.get_ylim()[1]*1.05, s='int')
                    ax.text(x=-2, y=ax.get_ylim()[1]*1.1, s='grouping')
                    ax.text(x=-2, y=ax.get_ylim()[1]*1.15, s=class_name)

                    for use_key in use_keys:
                        print(use_key)
                        grp_df = roi_df[roi_df['grouping'] == use_key]
                        # print(grp_df)
                        # pooling was performed across animals, so don't use LME, use ANOVA
                        dat = [grp_df.loc[grp_df['i_cls'] == i_cls, 'Accuracy'] for i_cls in range(len(class_labels))]
                        stat, pvals = stats.f_oneway(*dat)
                        print('stat = {:.4f}, pval = {:.4f}'.format(stat, pvals))
                        # print(stats.ttest_ind(*dat))  # equivalent to f_oneway above

                        # LME, for comparison
                        model = mixedlm('Accuracy ~ C(i_cls)', grp_df, groups='pop_id')
                        mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                        print(mfit.summary())
                        tmp = mfit.summary().tables[1][1:len(class_labels)]['P>|z|'].values
                        pvals = np.ones(len(class_labels) - 1) if np.all(tmp == '') else tmp.astype(np.float64)
                        print(pvals)
                        plot_stars(g.axes[i_roi, 0], np.arange(1, len(class_labels)), np.array(pvals), ytop_scale=1.15)

                        for i_cls, class_label in enumerate(class_labels):
                            # print(grp_df[grp_df[class_name] == class_label])
                            acc_df = grp_df.loc[grp_df[class_name] == class_label, 'Accuracy']
                            stat, pval = stats.ttest_1samp(acc_df, popmean=0)
                            # model = mixedlm('Accuracy ~ 1', grp_df[grp_df[class_name] == class_label], groups='pop_id')
                            # mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                            # print(mfit.summary())
                            # tmp = mfit.summary().tables[1]['P>|z|'][0]
                            # pval = 1 if tmp == '' else float(tmp)
                            print(class_label, '1 samp pval', pval)
                            intps[class_label].append(pval)
                            print('mean = ', np.mean(acc_df))
                            print('sem = ', stats.sem(acc_df))

                    if cls_flag:
                        plot_stars(g.axes[i_roi, 0], np.arange(len(class_labels)), [x[0] for x in intps.values()], ytop_scale=1.05)
                    else:
                        for i_cls, class_label in enumerate(class_labels):
                            plot_stars(g.axes[i_roi, i_cls], np.arange(len(use_keys)), intps[class_label], ytop_scale=1.05)

                    if not cls_flag:
                        for i_cls, class_label in enumerate(class_labels):
                            cls_df = roi_df[roi_df[class_name] == class_label]
                            # print(cls_df)

                            if len(use_keys) == 2:
                                dat = [cls_df.loc[cls_df['i_grouping'] == i_grp, 'Accuracy'] for i_grp in range(len(use_keys))]
                                stat, pvals = stats.ttest_rel(*dat)
                                print(class_label, 'stat = {:.4f}, pval = {:.4f}'.format(stat, pvals))

                            anova_res = AnovaRM(data=cls_df, depvar='Accuracy', subject='pop_id', within=['i_grouping']).fit()
                            print(class_label)
                            print(anova_res)
                            pvals = anova_res.anova_table["Pr > F"][0]
                            print(pvals)
                            # plot_stars(g.axes[i_roi, i_cls], [(len(use_keys) - 1) / 2], np.array([pvals]), ytop_scale=1.1)

                            # LME, for comparison
                            model = mixedlm('Accuracy ~ C(i_grouping)', roi_df[roi_df[class_name] == class_label], groups='pop_id', re_formula='~i_grouping')
                            mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                            print(mfit.summary())
                            pvals = mfit.summary().tables[1][1:len(use_keys)]['P>|z|'].values.astype(np.float64)
                            print(pvals)
                            plot_stars(g.axes[i_roi, i_cls], np.arange(1, len(use_keys)), np.array(pvals), ytop_scale=1.1)


                plt.savefig(save_dir + '{}_within_mouse_False_n_train_{}_C_{}_{}_{}_across_{}_per_{}_pop_{}_{}.pdf'.format(
                    grp['name'], n_train, reg_C, df_id, '_'.join(rois), class_name, per_key, use_pop, activity_type), bbox_inches='tight')


def get_sids(neuron_info):
    _, fig_path_inds = np.unique(neuron_info['fig_paths'], return_index=True)
    nsids = len(fig_path_inds)
    sid_lens = np.diff(np.insert(fig_path_inds, nsids, neuron_info.shape[0]))
    all_sids = np.repeat(np.arange(nsids), sid_lens)
    return all_sids


def compute_parallelism_split_interaction_by_class(class_name, class_labels, neuron_info, cue_resps, periods, rois=['All Subregions'],
                                                   metric='cosine', use_size=100, pseudo=False, by_mouse=True, n_splits=5):
    """
    :param class_name e.g. 'genotype', 'lesion'
    :param class_labels e.g. ['matched', 'lesioned']
    :param neuron_info: pandas DataFrame
    :param cue_resps: array shape (n_trial_types, n_neurons, max_n_trials_per_type, n_periods)
    :param periods: dictionary
    :param rois: list
    :param metric: string to input to sklearn
    :param use_size: number of neurons per mouse-region
    Shape = (n_shuffles, n_trial_types, n_neurons, n_periods)
    :return:
    """

    # parallelism score setup
    # ps_keys = ['B1-C1v.B2-C2', 'B1-C2v.B2-C1', 'B1-B2v.C1-C2']
    ps_keys = ['Variance direction 1', 'Variance direction 2']  #, 'Odor direction']
    pair_order = [(2, 4, 3, 5), (2, 5, 3, 4)]  # , (2, 3, 4, 5)]

    # types_to_pair = np.array([2, 3, 4, 5])

    dat = np.nanmean(cue_resps, axis=2)  # mean across trials, making it shape (n_trial_types, n_neurons, n_periods)
    rng = np.random.default_rng(seed=1)

    ps_resps = {class_label: {} for class_label in class_labels}
    # axis_sims = {class_label: {} for class_label in class_labels}
    # random_sims = {class_label: {} for class_label in class_labels}

    assert by_mouse or pseudo

    for i_label, class_label in enumerate(class_labels):
        print(class_label)
        mice = np.unique(neuron_info.loc[neuron_info[class_name] == class_label, 'names']) if by_mouse else np.arange(n_splits)
        ps_resps[class_label] = {roi: {mouse: {k: {per: {} for per in range(periods['n_prerew_periods'])} for k in ps_keys}
                                           for mouse in mice} for roi in rois}
        # axis_sims[class_label] = {
        #     roi: {mouse: {per: {} for per in range(periods['n_prerew_periods'])} for mouse in mice} for roi in rois}
        # random_sims[class_label] = {
        #     roi: {mouse: {per: {} for per in range(periods['n_prerew_periods'])} for mouse in mice} for roi in rois}

        for i_roi, roi in enumerate(rois):
            for i_mouse, mouse_name in enumerate(mice):
                # print(i_mouse)
                if by_mouse:
                    fig_paths = np.unique(neuron_info.loc[np.logical_and(neuron_info['names'] == mouse_name,
                                                                         neuron_info[class_name] == class_label), 'fig_paths'])
                else:
                    fig_paths = np.unique(neuron_info.loc[neuron_info[class_name] == class_label, 'fig_paths'])

                if pseudo:
                    inds = np.logical_and(neuron_info['names'] == mouse_name, neuron_info[class_name] == class_label) if by_mouse else neuron_info[class_name] == class_label
                    if roi != 'All Subregions': inds = np.logical_and(inds, neuron_info['str_regions'] == roi)

                    if inds.sum() >= use_size:
                        n_trials_per_cell = np.sum(~np.isnan(cue_resps[..., 0]), axis=2)  # (n_trial_types, n_neurons)
                        resps_per_split = np.zeros((dat.shape[0], dat.shape[1], n_splits, dat.shape[2])) # (n_trial_types, n_neurons, n_splits, n_periods)
                        for i_tt in np.arange(n_trials_per_cell.shape[0]):  # split trials of each trial type into disjoint subsets, as in decoding
                            for fig_path in fig_paths:
                                cell_inds = neuron_info['fig_paths'] == fig_path
                                trial_splits_per_cell = np.array_split(rng.permutation(np.arange(n_trials_per_cell[i_tt, np.flatnonzero(cell_inds)[0]])), n_splits)
                                for i_split in np.arange(n_splits):
                                    resps_per_split[i_tt, cell_inds, i_split] = np.mean(  # average over trials of that split
                                        np.take_along_axis(cue_resps[i_tt, cell_inds],
                                                           trial_splits_per_cell[i_split][np.newaxis, :, np.newaxis], axis=1), axis=1)

                        for i_split in np.arange(n_splits):  # resps_per_split[:, :, i_split] is same shape as dat
                            # ps_resps, axis_sims, random_sims = do_pairwise_dists(
                            #     inds, resps_per_split[:, :, i_split], rng, periods, pair_order, ps_resps, axis_sims,
                            #     random_sims, class_label, roi, mouse_name, ps_keys, use_size, metric, fig_path='pseudo_{}'.format(i_split))
                            ps_resps = do_pairwise_dists(inds, resps_per_split[:, :, i_split], rng, periods, pair_order,
                                                         ps_resps, class_label, roi, mouse_name, ps_keys, use_size, metric, fig_path='pseudo_{}'.format(i_split))

                    if not by_mouse:
                        break

                else:
                    # print(fig_paths)
                    for fig_path in fig_paths:
                        inds = neuron_info['fig_paths'] == fig_path
                        if roi != 'All Subregions':
                            inds = np.logical_and(inds, neuron_info['str_regions'] == roi)
                        # print(fig_path)
                        # ps_resps, axis_sims, random_sims = do_pairwise_dists(
                        #     inds, dat, rng, periods, pair_order, ps_resps, axis_sims, random_sims, class_label, roi,
                        #     mouse_name, ps_keys, use_size, metric, fig_path)
                        ps_resps = do_pairwise_dists(inds, dat, rng, periods, pair_order, ps_resps, class_label, roi,
                                                     mouse_name, ps_keys, use_size, metric, fig_path)

    return ps_resps  # , axis_sims, random_sims


def do_pairwise_dists(inds, dat, rng, periods, pair_order, ps_resps, class_label, roi,
                      mouse_name, ps_keys, use_size, metric, fig_path='pseudo'):

    for per in range(periods['n_prerew_periods']):
        if np.sum(inds) >= use_size:
            for i_po, po in enumerate(pair_order):
                use_dict = ps_resps[class_label][roi][mouse_name][ps_keys[i_po]][per]
                use_dict[fig_path] = np.zeros(use_size)
                for i_sub in range(use_size):  # use many subsamples, then take the mean
                    sub_inds = rng.choice(np.flatnonzero(inds), use_size, replace=False)
                    use_dict[fig_path][i_sub] = 1 - pairwise_distances(
                        np.vstack((dat[po[0], sub_inds, per] - dat[po[1], sub_inds, per],
                                   dat[po[2], sub_inds, per] - dat[po[3], sub_inds, per])),
                        metric=metric)[0, 1]
                use_dict[fig_path] = np.mean(use_dict[fig_path])

    return ps_resps  # , axis_sims, random_sims


def make_parallelism_interaction_df_by_class(class_name, use_resps, periods, pseudo=False):
    use_dict = {
        'grouping': [],
        'period': [],
        class_name: [],
        'Subregion': [],
        'Parallelism': [],
        'mouse': [],
        'pop_id': []}

    i_pop_id = 0
    for class_label in use_resps.keys():
        for roi in use_resps[class_label].keys():
            for mouse in use_resps[class_label][roi].keys():
                for key in use_resps[class_label][roi][mouse].keys():
                    if use_resps[class_label][roi][mouse][key]:
                        for per in range(periods['n_prerew_periods']):
                            ps_arr = list(use_resps[class_label][roi][mouse][key][per].values())
                            n_splits = len(ps_arr)
                            use_dict['Parallelism'].extend(ps_arr)
                            use_dict['period'].extend([per] * n_splits)
                            use_dict['grouping'].extend([key] * n_splits)
                            use_dict[class_name].extend([class_label] * n_splits)
                            use_dict['Subregion'].extend([roi] * n_splits)
                            use_dict['mouse'].extend([mouse] * n_splits)
                            use_dict['pop_id'].extend(np.arange(i_pop_id, i_pop_id + n_splits)) if pseudo else \
                                use_dict['pop_id'].extend(use_resps[class_label][roi][mouse][key][per].keys())
                            i_pop_id += n_splits

    return use_dict


def reduce(ret_df, neuron_info, X_means_norm, class_name, protocol_info, colors, late_trace_ind=3, n_components=2, activity_type='spks'):

    reducer = PCA(n_components=n_components)
    # reducer = FastICA(n_components=2)
    # reducer = FactorAnalysis(n_components=n_components)
    # reducer = NMF(n_components=n_components, max_iter=2000)

    n_cols = 6
    n_rows = int(np.ceil(len(ret_df) / n_cols))

    n_trace_types = protocol_info['n_trace_types']
    protocol = protocol_info['protocol']

    rda = np.zeros((len(ret_df), n_trace_types, n_trace_types))
    rda_metric = 'cosine'

    pairwise_dists = np.full((len(ret_df), n_components, n_trace_types, n_trace_types), np.nan)
    session_avg_components = {}
    explained_var = {}

    # loop over sessions
    fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_components > 2:
        fig3, axs3 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), subplot_kw=dict(projection="3d"))
    # fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    # fig3, axs3 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    ret = [plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows)) for _ in range(n_components)]
    figcomps = [x[0] for x in ret]; axscomps = [x[1] for x in ret]
    # print(figcomps)
    # print(axscomps)
    fig4, axs4 = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    sess_class_labels = np.empty(len(ret_df), dtype=object)

    for index, row in ret_df.iterrows():

        # Do dimensionality reduction on trial-type-averages
        cell_inds = np.logical_and(neuron_info['fig_paths'] == row['figure_path'], ~neuron_info[class_name].isnull())
        sess_class_label = np.unique(neuron_info.loc[cell_inds, class_name])
        assert len(sess_class_label) == 1
        sess_class_labels[index] = sess_class_label[0]
        # use normalized version!
        dat = X_means_norm[:n_trace_types, cell_inds, late_trace_ind]
        projection = reducer.fit_transform(dat)  # - dat.min())
        session_avg_components[index] = reducer.components_
        explained_var[index] = reducer.explained_variance_ratio_

        rda[index] = pairwise_distances(dat, metric=rda_metric)
        axs4.flat[index].pcolormesh(rda[index])

        for i_comp, axs in zip(np.arange(n_components), [ax.flat for ax in axscomps]):
            pairwise_dists[index, i_comp] = pairwise_distances(projection[:, i_comp].reshape(-1, 1), metric='euclidean')
            axs[index].pcolormesh(pairwise_dists[index, i_comp])
            axs[index].set_title(' '.join([row['name'], row['file_date_id'], sess_class_labels[index][0]]))

        ax = axs1.flat[index]
        # ax.scatter(projection[:, 0], projection[:, 1], color=colors['colors'][:n_trace_types])
        pdf = pd.DataFrame({'projection1': projection[:, 0],
                            'projection2': projection[:, 1],
                            'tt': np.arange(n_trace_types)})
        sns.scatterplot(pdf, x='projection1', y='projection2', hue='tt',
                        palette=list(colors['colors'][:n_trace_types]), s=100, ax=ax, legend=False)

        ax.set_ylim(np.array(ax.get_ylim()) * 1.2)
        ax.set_xlim(np.array(ax.get_xlim()) * 1.2)
        ax.set_title(' '.join([row['name'], row['file_date_id'], sess_class_labels[index][0]]))
        ax.set_xlabel('')
        ax.set_ylabel('')

        if n_components > 2:
            ax = axs3.flat[index]
            ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], color=colors['colors'][:n_trace_types], s=100)

    for i_fig, fig in enumerate([fig1, *figcomps]):
        fig.tight_layout()
        hide_spines()
        fig.savefig('../neural-plots/sessionwise_pca_{}_{}_{}_fig_{}.pdf'.format(protocol, class_name, activity_type, i_fig), bbox_inches='tight')

    return pairwise_dists, rda, session_avg_components, explained_var, sess_class_labels


def avg_session_dists(ret_df, pairwise_dists, rda, class_labels, n_trace_types, sess_class_labels, n_components=2):

    mouse_names, unique_sids = np.unique(ret_df['name'], return_index=True)
    n_mice = len(mouse_names)
    mouse_dists = np.full((n_mice, len(class_labels), n_components, n_trace_types, n_trace_types), np.nan)
    mouse_rda = np.full((n_mice, len(class_labels), 1, n_trace_types, n_trace_types), np.nan)
    for i_mouse, mouse_name in enumerate(mouse_names):
        for i_cl, class_label in enumerate(class_labels):
            sid_inds = np.logical_and(ret_df['name'] == mouse_name, sess_class_labels == class_label)
            mouse_dists[i_mouse, i_cl] = np.mean(pairwise_dists[sid_inds], axis=0)
            mouse_rda[i_mouse, i_cl] = np.mean(rda[sid_inds], axis=0)

    avg_dists = np.nanmean(mouse_dists, axis=0)
    avg_rda = np.nanmean(mouse_rda, axis=0)
    # tt_labels = np.array(['Nothing 1', 'Nothing 2', 'Fixed 1', 'Fixed 2', 'Variable 1', 'Variable 2'], dtype='object')

    return avg_dists, avg_rda


def project(class_name, class_label, i_cls, iter_name, iter_i, iter_label, n_trace_types, dat, rda, pca_dict, var_dict,
            dists, axs, axs3, colors, rda_metric='cosine', n_components=2):
    rda[iter_i, i_cls, :, :] = pairwise_distances(dat, metric=rda_metric)

    pca = PCA(n_components=n_trace_types)
    # pca = FastICA(n_components=2)
    projection = pca.fit_transform(dat)
    pca_dict[class_name].extend([class_label] * n_trace_types)
    pca_dict[iter_name].extend([iter_label] * n_trace_types)
    pca_dict['tt'].extend(np.arange(n_trace_types))
    pca_dict['pc1'].extend(projection[:, 0])
    pca_dict['pc2'].extend(projection[:, 1])
    pca_dict['components'].extend(pca.components_.tolist())

    # previously, I had combined pca_dict and var_dict, but the semantics are actually different (i.e. trial type != component),
    # so keep them separate
    var_dict[class_name].extend([class_label] * n_trace_types)
    var_dict[iter_name].extend([iter_label] * n_trace_types)
    var_dict['component'].extend(np.arange(n_trace_types))
    var_dict['explained_var'].extend(pca.explained_variance_ratio_)

    for i_comp in np.arange(n_components):
        dists[iter_i, i_cls, i_comp] = pairwise_distances(projection[:, i_comp].reshape(-1, 1))

    ax = axs[iter_i, i_cls]
    # ax.scatter(projection[:, 0], projection[:, 1], s=60, color=colors['colors'][:n_trace_types])
    pdf = pd.DataFrame({'projection1': projection[:, 0],
                        'projection2': projection[:, 1],
                        'tt': np.arange(n_trace_types)})
    sns.scatterplot(pdf, x='projection1', y='projection2', hue='tt',
                    palette=list(colors['colors'][:n_trace_types]), s=100, ax=ax, legend=False)
    ax.set_xlim(np.array(ax.get_xlim()) * 1.2)
    ax.set_ylim(np.array(ax.get_ylim()) * 1.2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['left'].set_position(("axes", -0.1))
    if iter_i == 0: ax.set_title(class_label)

    ax = axs3[iter_i, i_cls]
    ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], s=100, color=colors['colors'][:n_trace_types])
    if iter_i == 0: ax.set_title(class_label)

    return rda, pca_dict, dists, projection, var_dict


def plot_scree(dists, rda, pca, class_name, class_labels, class_colors, prefix, protocol):
    dists_avg = np.nanmean(dists, axis=0)
    rda_avg = np.nanmean(rda, axis=0)

    pca_df = pd.DataFrame(pca)
    plt.figure()
    sns.lineplot(data=pca_df, x='component', y='explained_var', hue=class_name, hue_order=class_labels,
                 palette=class_colors)
    plt.ylabel('Explained variance')
    hide_spines()
    plt.savefig('../neural-plots/{}_{}_pca_scree_{}.pdf'.format(protocol, prefix, class_name), bbox_inches='tight')

    return dists_avg, rda_avg


def reduce_pool_mice(neuron_info, use_cue_resps, class_name, class_labels, protocol_info, colors, class_colors,
                     late_trace_ind=3, n_components=2, n_splits=5, activity_type='spks'):

    n_trace_types = protocol_info['n_trace_types']
    protocol = protocol_info['protocol']

    # Also try doing PCA on all neurons from all mouse
    pca = PCA(n_components=n_trace_types)
    # pca = FastICA(n_components=2)
    iter_name = 'i_split'
    pooled_pca = {class_name: [], iter_name: [], 'tt': [], 'pc1': [], 'pc2': [], 'components': []}
    pooled_var = {class_name: [], iter_name: [], 'component': [], 'explained_var': []}
    pooled_dists = np.full((n_splits, len(class_labels), n_components, n_trace_types, n_trace_types), np.nan)

    pooled_rda = np.zeros((n_splits, len(class_labels), 1, n_trace_types, n_trace_types))
    rda_metric = 'cosine'

    pooled_fig, pooled_axs = plt.subplots(1, len(class_labels), figsize=(len(class_labels) * 2.2, 2), squeeze=False)
    split_fig, split_axs = plt.subplots(n_splits, len(class_labels), figsize=(len(class_labels) * 2.2, n_splits * 2), squeeze=False)
    [fig.tight_layout() for fig in [pooled_fig, split_fig]]

    pooled3_fig, pooled3_axs = plt.subplots(1, len(class_labels), figsize=(len(class_labels) * 2.2, 2), squeeze=False,
                                            subplot_kw=dict(projection="3d"))
    split3_fig, split3_axs = plt.subplots(n_splits, len(class_labels), figsize=(len(class_labels) * 6, 6), squeeze=False,
                                          subplot_kw=dict(projection="3d"))

    for i_cls, class_label in enumerate(class_labels):
        print(class_label)
        class_inds = neuron_info[class_name] == class_label
        for i_split in range(n_splits + 1):
            if i_split < n_splits:
                dat = np.nanmean(use_cue_resps[:n_trace_types, :, i_split::n_splits, late_trace_ind], axis=2)
                dat = dat[:, class_inds]  # need to do this in two steps to keep the shape
                pooled_rda, pooled_pca, pooled_dists, _, pooled_var = project(
                    class_name, class_label, i_cls, iter_name, i_split, i_split, n_trace_types, dat, pooled_rda,
                    pooled_pca, pooled_var, pooled_dists, split_axs, split3_axs, colors, rda_metric, n_components)

            else:
                dat = np.nanmean(use_cue_resps[:n_trace_types, :, :, late_trace_ind], axis=2)
                dat = dat[:, class_inds]
                projection = pca.fit_transform(dat)
                components = pca.components_
                print('computed projection')

        ax = pooled_axs[0, i_cls]
        # ax.scatter(projection[:, 0], projection[:, 1], s=60, color=colors['colors'][:n_trace_types])
        pdf = pd.DataFrame({'projection1': projection[:, 0],
                            'projection2': projection[:, 1],
                            'tt': np.arange(n_trace_types)})
        sns.scatterplot(pdf, x='projection1', y='projection2', hue='tt',
                        palette=list(colors['colors'][:n_trace_types]), s=100, ax=ax, legend=False)
        ax.set_xlim(np.array(ax.get_xlim()) * 1.2)
        ax.set_ylim(np.array(ax.get_ylim()) * 1.2)
        ax.set_title(class_label)
        # ax.set_ylabel('')
        # ax.set_xlabel('')
        ax.spines['left'].set_position(("axes", -0.1))

        ax = pooled3_axs[0, i_cls]
        ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], s=100, color=colors['colors'][:n_trace_types])
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

    hide_spines()
    split_fig.savefig('../neural-plots/split_pca_{}_{}_{}.pdf'.format(protocol, class_name, activity_type), bbox_inches='tight')
    pooled_fig.savefig('../neural-plots/pooled_pca_{}_{}_{}.pdf'.format(protocol, class_name, activity_type), bbox_inches='tight')

    pooled_avg, pooled_rdaavg = plot_scree(pooled_dists, pooled_rda, pooled_var, class_name, class_labels, class_colors, 'pooled', protocol)

    return pooled_pca, pooled_dists, pooled_avg, pooled_rda, pooled_rdaavg, pooled_var


def reduce_mouse_avg(neuron_info, X_means_norm, class_name, class_labels, mice, protocol_info, colors, class_colors,
                     late_trace_ind=3, n_components=2, min_size=0):
    # Also try doing PCA on all neurons from a mouse (that is, pooling across sessions within mice to increase
    # statistical power that way)
    n_trace_types = protocol_info['n_trace_types']
    protocol = protocol_info['protocol']

    n_mice = len(mice)
    mousewise_pca = {class_name: [], 'mouse': [], 'tt': [], 'pc1': [], 'pc2': [], 'components': []}
    mousewise_var = {class_name: [], 'mouse': [], 'component': [], 'explained_var': []}
    mousewise_dists = np.full((n_mice, len(class_labels), n_components, n_trace_types, n_trace_types), np.nan)
    mousewise_rda = np.full((n_mice, len(class_labels), 1, n_trace_types, n_trace_types), np.nan)
    rda_metric = 'cosine'

    fig, axs = plt.subplots(n_mice, len(class_labels), figsize=(len(class_labels) * 2, len(mice) * 2), squeeze=False)
    #                        sharex=True, sharey=True)
    fig.tight_layout()
    fig3, axs3 = plt.subplots(n_mice, len(class_labels), figsize=(len(class_labels) * 4, len(mice) * 4), squeeze=False, subplot_kw=dict(projection="3d"))

    for i_cls, class_label in enumerate(class_labels):
        for i_mouse, mouse_name in enumerate(mice):

            mouse_class_inds = np.logical_and(neuron_info[class_name] == class_label, neuron_info['names'] == mouse_name)
            if np.sum(mouse_class_inds) > max(n_trace_types, min_size):

                dat = X_means_norm[:n_trace_types, mouse_class_inds, late_trace_ind]
                mousewise_rda, mousewise_pca, mousewise_dists, projection, mousewise_var = project(
                    class_name, class_label, i_cls, 'mouse', i_mouse, mouse_name, n_trace_types, dat, mousewise_rda,
                    mousewise_pca, mousewise_var, mousewise_dists, axs, axs3, colors, rda_metric, n_components)

    hide_spines()
    plt.savefig('../neural-plots/mousewise_pca_{}_{}.pdf'.format(protocol, class_name), bbox_inches='tight')

    mousewise_avg, mousewise_rdaavg = plot_scree(mousewise_dists, mousewise_rda, mousewise_var, class_name, class_labels, class_colors, 'mousewise', protocol)

    return mousewise_pca, mousewise_dists, mousewise_avg, mousewise_rda, mousewise_rdaavg, mousewise_var


def reduce_stats(pairwise_dists, mousewise_dists, pooled_dists, rda, mousewise_rda, pooled_rda, ret_df, dec_dict,
                 protocol_info, class_name, class_labels, sess_class_labels, mice, mouse_colors,
                 n_components=2, trunc_df=None, trunc_class_labels=None, n_splits=5, activity_type='spks', behave=False):

    n_trace_types = protocol_info['n_trace_types']
    protocol = protocol_info['protocol']
    tt_labels = np.array(protocol_info['trace_type_names'], dtype='object')

    # plot class_labels together (for interaction) as well as separately (for main effects):
    iter_class_labels = [[x] for x in class_labels] + [class_labels] if len(class_labels) > 1 else [class_labels]
    for i_cl, use_class_labels in enumerate(iter_class_labels):

        # print(use_class_labels)

        # n_mice = len(mice)
        which_sess = np.ones(len(sess_class_labels), dtype=bool) if np.all(sess_class_labels == 'all') else np.isin(sess_class_labels, use_class_labels)
        which_mice = np.unique(ret_df.loc[which_sess, 'name'])
        # print(which_mice)
        use_mice = np.isin(mice, which_mice)
        # print(use_mice)
        # use_splits = np.ones(n_splits, dtype=bool) if i_cl < len(class_labels) else np.ones(2 * n_splits, dtype=bool)
        n_mice = len(which_mice)
        n_cl = len(use_class_labels)

        if trunc_df is None:
            trunc_df = ret_df
        if trunc_class_labels is None:
            trunc_class_labels = sess_class_labels

        mask = np.tri(n_trace_types, k=0)
        n_unique_pairs = (n_trace_types ** 2 - n_trace_types) / 2
        comp_order = ['{} vs. {}'.format(x, y) for i, x in enumerate(tt_labels) for j, y in enumerate(tt_labels) if
                      i < j]
        within_mean_keys = ['Fixed 1 vs. Fixed 2', 'Fixed 1 vs. Variable 1', 'Fixed 1 vs. Variable 2',
                            'Fixed 2 vs. Variable 1', 'Fixed 2 vs. Variable 2', 'Variable 1 vs. Variable 2',
                            'Uniform 1 vs. Uniform 2', 'Bimodal 1 vs. Bimodal 2', 'Uniform 1 vs. Bimodal 1',
                            'Uniform 1 vs. Bimodal 2', 'Uniform 2 vs. Bimodal 1', 'Uniform 2 vs. Bimodal 2']
        within_dist_keys = ['Fixed 1 vs. Fixed 2', 'Variable 1 vs. Variable 2',
                            'Uniform 1 vs. Uniform 2', 'Bimodal 1 vs. Bimodal 2']
        nothing_vs_fixed_keys = ['Nothing 1 vs. Fixed 1', 'Nothing 1 vs. Fixed 2', 'Nothing 2 vs. Fixed 1',
                                 'Nothing 2 vs. Fixed 2', 'Nothing 1 vs. Uniform 1', 'Nothing 1 vs. Uniform 2',
                                 'Nothing 2 vs. Uniform 1', 'Nothing 2 vs. Uniform 2']
        nothing_vs_variable_keys = ['Nothing 1 vs. Variable 1', 'Nothing 1 vs. Variable 2', 'Nothing 2 vs. Variable 1',
                                    'Nothing 2 vs. Variable 2', 'Nothing 1 vs. Bimodal 1', 'Nothing 1 vs. Bimodal 2',
                                    'Nothing 2 vs. Bimodal 1', 'Nothing 2 vs. Bimodal 2']
        # behave = True if 'behavior' in dec_dict['per']['ccgp']['scores'][use_class_labels[0]].keys() else False

        for dist_name, dist, n_comps in zip(['pca', 'mouse_pca', 'pooled_pca', 'rda', 'mouse_rda', 'pooled_rda'],
                                            [pairwise_dists, mousewise_dists, pooled_dists, rda, mousewise_rda, pooled_rda],
                                            [n_components, n_components, n_components, 1, 1, 1]):

            use_mouse_colors = mouse_colors
            if dist_name == 'pca' or dist_name == 'rda':
                pair_dict = {'mouse': np.repeat(trunc_df['name'], n_comps * n_unique_pairs),
                             'fig_path': np.repeat(trunc_df['figure_path'], n_comps * n_unique_pairs),
                             class_name: np.repeat(trunc_class_labels, n_comps * n_unique_pairs),
                             'component': np.tile(np.repeat(np.arange(n_comps), n_unique_pairs), len(trunc_df)),
                             'comparison': np.tile(np.arange(n_unique_pairs, dtype=np.int16), n_comps * len(trunc_df)),
                             'comp_order': np.tile(comp_order, n_comps * len(trunc_df)),
                             'dist': dist.reshape((len(trunc_df), n_comps, -1))[:, :,
                                     ~mask.astype(bool).flatten()].flatten()
                             }
                use_rois = ['All Subregions']
            else:
                use_rois = use_class_labels if class_name == 'str_regions' else ['All Subregions']
                if 'mouse' in dist_name:
                    n_use = n_mice
                    n_tile = n_use
                    which_use = use_mice
                    which_names = which_mice
                    # print(len(which_names), n_use, n_cl, n_comps, n_unique_pairs, n_tile)

                    pair_dict = {'mouse': np.repeat(which_names, n_cl * n_comps * n_unique_pairs),
                                 'fig_path': np.repeat(which_names, n_cl * n_comps * n_unique_pairs),
                                 class_name: np.tile(np.repeat(use_class_labels, n_comps * n_unique_pairs), n_tile),
                                 'component': np.tile(np.repeat(np.arange(n_comps), n_unique_pairs), n_cl * n_tile),
                                 'comparison': np.tile(np.arange(n_unique_pairs, dtype=np.int16),
                                                       n_comps * n_cl * n_tile),
                                 'comp_order': np.tile(comp_order, n_comps * n_cl * n_tile)
                                 }

                elif 'pooled' in dist_name:
                    n_use = n_splits
                    which_use = np.ones(n_splits, dtype=bool)
                    which_names = np.arange(n_splits)

                    if class_name == 'genotype':
                        n_tot = n_splits * len(class_labels)
                        # use_mouse_colors = {k: v for k, v in zip(np.arange(n_tot), sns.husl_palette(n_tot))}  # , h=i_cl/n_tot)[::len(class_labels)])}
                        if i_cl < len(iter_class_labels) - 1:
                            which_names = np.arange(i_cl, n_tot, 2)
                            # which_names = np.arange(n_splits * i_cl, n_splits * (i_cl + 1))

                    else:
                        n_tot = n_splits * len(use_class_labels)
                        # use_mouse_colors = {k: v for k, v in zip(np.arange(n_use), sns.husl_palette(n_use))}

                    if i_cl == len(iter_class_labels)  - 1:
                        which_names = np.arange(n_tot)
                        # which_names = np.array(list(np.arange(0, n_tot, 2)) + list(np.arange(1, n_tot, 2)))

                    use_mouse_colors = {k: v for k, v in zip(np.arange(n_tot), sns.husl_palette(n_tot))}  # , h=i_cl/n_tot)[::len(class_labels)])}
                    # print(len(which_names), n_use, n_cl, n_comps, n_unique_pairs)

                    pair_dict = {'mouse': np.repeat(which_names, n_comps * n_unique_pairs),
                                 'fig_path': np.repeat(which_names, n_comps * n_unique_pairs),
                                 class_name: np.tile(np.repeat(use_class_labels, n_comps * n_unique_pairs), n_use),
                                 'component': np.tile(np.repeat(np.arange(n_comps), n_unique_pairs), n_cl * n_use),
                                 'comparison': np.tile(np.arange(n_unique_pairs, dtype=np.int16),
                                                       n_comps * n_cl * n_use),
                                 'comp_order': np.tile(comp_order, n_comps * n_cl * n_use)
                                 }

                if i_cl < len(class_labels):
                    pair_dict['dist'] = dist[which_use, i_cl].reshape((n_use, n_cl, n_comps, -1))[:, :, :,
                                        ~mask.astype(bool).flatten()].flatten()
                else:
                    pair_dict['dist'] = dist.reshape((n_use, n_cl, n_comps, -1))[:, :, :,
                                        ~mask.astype(bool).flatten()].flatten()

            # [print(k, len(v)) for k, v in pair_dict.items()]
            pair_df = pd.DataFrame(pair_dict).dropna()
            if dist_name == 'pca' or dist_name == 'rda':
                pair_df = pair_df[np.isin(pair_df['mouse'], which_mice)]
            # print(np.unique(pair_df[class_name]))
            # print(pair_df)
            # print(use_mouse_colors)
            for key, label in zip(['mean', 'pair', 'odor'], ['mean', 'distribution', '']):  # ['odor2'], ['']):

                print(dist_name, key, use_class_labels)

                grouping = 'Across ' + label
                comp_df = pair_df.copy()
                # print(comp_df)
                comp_constants = {'grouping': grouping, 'Subregion': 'All Subregions', 'period': '3_3'}
                comp_df = comp_df.assign(**comp_constants)
                ylabel = 'Representational dissimilarity'
                use_colors = dec_dict['per'][key]['pooled_colors']

                if key == 'mean':

                    if 'pca' in dist_name:
                        comp_df = comp_df[comp_df['component'] == 0]  # distance along pc 1
                        ylabel = 'Distance along PC 1 (a.u.)'
                    is_within_mean = np.isin(comp_df['comp_order'], within_mean_keys)
                    comp_df.loc[is_within_mean, 'grouping'] = 'Within mean'

                elif key == 'pair':

                    comp_df = comp_df[~comp_df['comp_order'].str.contains('Nothing').values]
                    if 'pca' in dist_name:
                        comp_df = comp_df[comp_df['component'] == 1]  # distance along pc 2
                        ylabel = 'Distance along PC 2 (a.u.)'
                    is_within_dist = np.isin(comp_df['comp_order'], within_dist_keys)
                    comp_df.loc[is_within_dist, 'grouping'] = 'Within distribution'

                elif 'odor' in key:
                    if 'pca' in dist_name:
                        if '2' in key:
                            comp_df = comp_df[comp_df['component'] == 1]  # distance along pc 2
                            ylabel = 'Distance along PC 2 (a.u.)'
                        else:
                            comp_df = comp_df[comp_df['component'] == 0]  # distance along pc 1
                            ylabel = 'Distance along PC 1 (a.u.)'

                    is_nothing_vs_fixed = np.isin(comp_df['comp_order'], nothing_vs_fixed_keys)
                    is_nothing_vs_variable = np.isin(comp_df['comp_order'], nothing_vs_variable_keys)
                    if 'Fixed 1' in tt_labels:
                        odor_labels = ['Nothing vs. Variable', 'Nothing vs. Fixed']
                        use_colors = {'Nothing vs. Fixed': '#d62728', 'Nothing vs. Variable': '#1f77b4'}
                    elif 'Uniform 1' in tt_labels:
                        odor_labels = ['Nothing vs. Bimodal', 'Nothing vs. Uniform']
                        use_colors = {'Nothing vs. Uniform': '#bb4513', 'Nothing vs. Bimodal': '#1f77b4'}
                    else:
                        raise Exception('tt_labels not recognized')

                    comp_df.loc[is_nothing_vs_fixed, 'grouping'] = odor_labels[1]
                    comp_df.loc[is_nothing_vs_variable, 'grouping'] = odor_labels[0]
                    comp_df = comp_df[np.isin(comp_df['grouping'], odor_labels)]
                    # print(len(comp_df))

                # print(np.unique(comp_df['mouse']))

                if class_name != 'str_regions' or len(use_class_labels) > 1:

                    labels = ['Across ' + label, 'Within ' + label] if label else odor_labels
                    tmp_class_labels = use_class_labels if 'mouse' in dist_name or 'pooled' in dist_name else np.unique(sess_class_labels)

                    g, stat_df = general_plotter(False, comp_df, class_name, tmp_class_labels, 'dist', 'fig_path', labels,
                                                 use_colors, ['All Subregions'], use_mouse_colors, None, within=True)

                    if 'pooled' in dist_name:
                        # print(comp_df)
                        # print(stat_df)
                        if len(use_class_labels) == 1:
                            stat, pval = stats.ttest_1samp(stat_df['dist'], popmean=0)
                        else:
                            stat, pval = stats.f_oneway(  # confirmed this is equivalent to statmodels anova_lm types 1-3
                                *[stat_df.loc[stat_df[class_name] == cl, 'dist'] for cl in use_class_labels])
                        print('stat = {:.4f}, pval = {:.4f}'.format(stat, pval))
                        # at some point, consider reverting so I plot the anova stars, but for now, plot the LME stars
                        # plot_stars(g.axes.flat[0], [0.5], [pval])

                    # else:
                    #     break

                    # elif 'mouse' in dist_name and len(use_class_labels) > 1:
                    #     print(AnovaRM(data=stat_df, depvar='dist', subject='mouse', within=[class_name]).fit())

                    # else: # do this always, for now, until I revert to potential ANOVA
                    # print(stat_df)
                    # this computes differences across subregions, relative to DLS
                    # if len(tmp_class_labels) <= 2:
                    if class_name == 'str_regions':
                        print(tmp_class_labels)
                        roi_keys = {k: tmp_class_labels.index(k) for k in tmp_class_labels}
                        roi_keys['VP'] = -1
                        stat_df['i_roi'] = stat_df[class_name].apply(lambda x: roi_keys[x])
                        _ = general_stats(False, stat_df, 'dist', g.axes.flat[0], 'All Subregions', 'i_roi',
                                          roi_keys.values(), 'lme', labels, 'pool')
                    else:
                        _ = general_stats(False, stat_df, 'dist', g.axes.flat[0], 'All Subregions', class_name,
                                          tmp_class_labels, 'lme', labels, 'pool')

                    # also compute differences within subregions/classes
                    if len(tmp_class_labels) > 1:
                        for i, (class_label, ax) in enumerate(zip(tmp_class_labels, g.axes.flat)):
                            class_df = stat_df[stat_df[class_name] == class_label]
                            if len(np.unique(class_df['mouse'])) > 1:
                                model = mixedlm('dist ~ 1', class_df, groups='mouse')
                                mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
                                print(class_label, mfit.summary())
                                tmp = mfit.summary().tables[1]['P>|z|'][0]
                                pval = 1 if tmp == '' else float(tmp)
                                plot_stars(ax, [.5], [pval], ytop_scale=0.9)

                    g.set(ylabel=ylabel, title=dist_name)
                    g.fig.suptitle(' '.join(use_class_labels), y=1.1)
                    if behave:
                        plt.savefig('../behavior-plots/{}_{}_{}_across_{}_{}_{}_behavior.pdf'.format(
                            protocol, dist_name, class_name, '_'.join(tmp_class_labels), key, label), bbox_inches='tight')
                    else:
                        plt.savefig('../neural-plots/{}_{}_{}_across_{}_{}_{}_{}.pdf'.format(
                            protocol, dist_name, class_name, '_'.join(tmp_class_labels), key, label, activity_type), bbox_inches='tight')


def plot_masked_mat(avg, i_comp, start_tt, class_labels, name, protocol_info):

    n_trace_types = protocol_info['n_trace_types']
    tt_labels = np.array(protocol_info['trace_type_names'], dtype='object')
    protocol = protocol_info['protocol']

    comp_data = avg[:, i_comp, start_tt:, start_tt:]
    # print(comp_data.shape)
    vmin = np.amin(comp_data[comp_data > 0], axis=None)
    vmax = np.amax(comp_data, axis=None)

    fig, axs = plt.subplots(1, len(class_labels), figsize=(len(class_labels) * 2, 2),
                            gridspec_kw={'wspace': .4}, squeeze=False)

    for i_cl, ax in enumerate(axs.flat):

        dists = avg[i_cl, i_comp, start_tt:, start_tt:]
        mask = np.tri(dists.shape[0], k=0)
        masked = np.ma.array(dists, mask=mask)

        im = ax.imshow(masked.T, vmin=vmin - (vmax - vmin) * .1, vmax=vmax, cmap=cmocean.cm.dense)
        #     plt.pcolormesh(np.arange(n_trace_types-start_tt), np.arange(n_trace_types-start_tt, 0, -1),
        #                    avg_dists[i_comp, start_tt:, start_tt:])
        if i_cl == 0:
            ax.set_yticks(np.arange(n_trace_types - start_tt))
            ax.set_yticklabels(tt_labels[start_tt:])
        ax.set_xticks(np.arange(n_trace_types - start_tt))
        ax.set_xticklabels(tt_labels[start_tt:], rotation=90)
        ax.set_title(class_labels[i_cl])

    if 'avg_rda' in name:
        title = 'Representational dissimilarity'
    else:
        title = 'Distance along PC {} (a.u.)'.format(i_comp + 1)
    add_cbar(fig, im, title, width=.03)
    #     plt.colorbar()
    hide_spines()
    save_dir = '../behavior-plots' if 'behavior' in name else '../neural-plots'
    plt.savefig(os.path.join(save_dir, 'pca_pairwise_mat_{}_{}_{}_{}_comp_{}.pdf'.format(
        protocol, '_'.join(class_labels), start_tt, name, i_comp + 1)), bbox_inches='tight')


def get_row_col_hue(rois, class_name, class_labels, use_keys, grping='grouping', style_order=['all']):

    # defaults
    row_name = 'Subregion'
    row_order = rois
    hue = grping
    hue_order = use_keys

    if len(rois) == 1:
        if len(use_keys) == 1:
            col_name = grping
            col_order = use_keys
            hue = class_name
            hue_order = class_labels
        elif len(style_order) > 1:
            # quit using style, and use cols instead
            col_name = 'dec_key'
            col_order = style_order
        else:
            col_name = class_name
            col_order = class_labels

    else:
        col_name = 'Subregion'
        col_order = rois
        row_name = class_name
        row_order = class_labels

    return row_name, row_order, col_name, col_order, hue, hue_order


def general_plotter(pseudo, sub_df, class_name, class_labels, depvar, strat, use_keys, use_colors, rois, mouse_colors,
                    class_colors, within=False, hlines=None, style_order=['all'], ylims=None, yticks=None, grping='grouping', unit='mouse'):
    """
    :param pseudo:
    :param sub_df:
    :param class_name:
    :param class_labels:
    :param depvar:
    :param strat:
    :param use_keys:
    :param use_colors:
    :param rois:
    :param mouse_colors:
    :param within:
    :param hlines:
    :param style_order:
    :param ylims:
    :param yticks:
    :param grp: This will be equal to 'grouping' in almost all cases. But when within and across-distribution aren't releveant
    (e.g. mean decoding), we'll use 'key' instead (e.g. Nothing vs. Variable)
    :return:
    """

    print(use_keys)

    if pseudo: assert not within  # only consider computing within-session differences if not pseudopopulations

    row_name, row_order, col_name, col_order, hue, hue_order = get_row_col_hue(rois, class_name, class_labels, use_keys, grping, style_order)
    print(row_name, row_order, col_name, col_order, hue, hue_order)
    palette = class_colors if hue == class_name else use_colors

    aspect = (len(use_keys) + 3) / 10 if len(use_keys) > 1 else .5
    grid_kwargs = dict(col=col_name, col_order=col_order, row=row_name, row_order=row_order, aspect=aspect,
                       height=2, sharex=False, gridspec_kws={'wspace': 0.1, 'hspace': 0.5})
    hue_kwargs = dict(x=hue, y=depvar, hue=unit, hue_order=mouse_colors.keys(), palette=mouse_colors, zorder=1,
                      legend=False)

    # mean_kwargs = dict(x=grp, y=depvar, hue=grp, order=use_keys, hue_order=use_keys, palette=use_colors,
    #                    color='k', errwidth=4, errorbar=('ci', 95))  # err_style='bars', err_kws={'lw': 4},zorder=50,
    mean_kwargs = dict(x=hue, y=depvar, hue=hue, order=hue_order, hue_order=hue_order, palette=palette,
                       color='k', errwidth=4, errorbar=('ci', 95))  # err_style='bars', err_kws={'lw': 4},zorder=50,

    # mean_kwargs = dict(x=grp, y=depvar, hue=grp, hue_order=use_keys, palette=use_colors, legend='brief',
    #                    estimator='mean', errorbar=('ci', 95), sort=False, err_style='bars', err_kws={'elinewidth': 4},
    #                    linestyle=None, marker='o', zorder=50)  # err_style='bars', err_kws={'lw': 4},zorder=50,

    # catplot_kwargs = dict(y=depvar, hue='mouse', x=x, hue_order=mouse_colors.keys(), kind='swarm',
    #                         palette=mouse_colors.values(), dodge=False, legend=False, sharey=True)
    #
    # boxplot_kwargs = dict(y=depvar, x=x, showmeans=True, meanline=True, meanprops={'color': 'k', 'ls': '-', 'lw': 3},
    #                       medianprops={'visible': False}, whiskerprops={'visible': False}, zorder=10, kind='box',
    #                       showfliers=False, showbox=False, showcaps=False, width=.5, legend=False, sharey=True)

    if not set(use_keys).isdisjoint(['odor', 'ccgp', 'var']):  # odor/ccgp decoding, just one pairing, so we'll look at intercept
        stat_df = sub_df.copy()
        stat_df[depvar] -= hlines  # subtract off chance to get difference from zero

    elif within:  # for simultaneous populations, consider compute differences within-session

        # average within x for each session separately
        # this is grping, not hue, b/c we'd never subtract off lesion/region/genotype
        # print(sub_df)
        # print(grping)
        # print(depvar)
        # print(class_name)
        pivot_df = sub_df.pivot_table(columns=grping, values=depvar,
                                      index=['mouse', strat, 'Subregion', class_name, 'period'])
        # print(pivot_df)
        # take the difference within session with respect to the last entry in use_keys
        pivot_diff = pivot_df[use_keys].sub(pivot_df[use_keys[-1]], axis=0).reset_index()
        # print(pivot_diff)
        # unpivot
        stat_df = pivot_diff.melt(id_vars=['mouse', strat, 'Subregion', class_name],
                                  value_vars=use_keys[:-1], var_name=grping, value_name=depvar).sort_values(
            by=grping, key=lambda series: [use_keys.index(x) for x in series]).reset_index()
        # print(melt_df)

    else:  # for pseudopopulations, we'll just look at across-grouping comparisons
        stat_df = sub_df

    if 'dec_key' not in sub_df:  # for decoder plots, we want to plot CCGP, pair, and cong on same axes
        sub_df['dec_key'] = 'all'
    # for plotting purposes, compute the mean difference across sessions within unit (usually mice, or pop_id in case of pooling across mice)
    agg_df = sub_df.groupby([unit, 'Subregion', class_name, grping, 'dec_key']).mean().sort_values(
        by=hue, key=lambda series: [hue_order.index(x) for x in series]).reset_index()

    g = sns.FacetGrid(data=agg_df, **grid_kwargs)
    if len(hue_order) == 1 or hue == 'genotype':
        g.map_dataframe(sns.swarmplot, size=8, **hue_kwargs)
        g.map_dataframe(sns.pointplot, scale=1.5, dodge=0.4, **mean_kwargs).set_titles("")  # "{col_name}")
    elif np.all(col_order == style_order) and col_order != ['all']:
        g.map_dataframe(sns.swarmplot, size=6, **hue_kwargs)
        g.map_dataframe(sns.pointplot, scale=1, dodge=0.4, **mean_kwargs).set_titles("")  # "{col_name}")
        g2 = sns.FacetGrid(data=agg_df, **grid_kwargs)
        g2.map_dataframe(sns.lineplot, estimator=None, **hue_kwargs)
        g2.map_dataframe(sns.pointplot, **mean_kwargs).set_titles("")
        g2.axes.flat[-1].legend(loc=(1.04, 0))
        g2.set(xlim=(-.5, len(hue_order) - .5))
        arrange_grid(g2, col_order, hue_order, hlines, ylims, yticks)
    else:
        g.map_dataframe(sns.lineplot, estimator=None, **hue_kwargs)  # style='dec_key', style_order=list(style_order),
        g.map_dataframe(sns.pointplot, **mean_kwargs).set_titles("")  # "{col_name}")

    g.axes.flat[-1].legend(loc=(1.04, 0))
    g.set(xlim=(-.5, len(hue_order) - .5))
    arrange_grid(g, col_order, hue_order, hlines, ylims, yticks)

    if np.all(col_order == style_order) and col_order != ['all']:
        g = [g, g2]
    return g, stat_df


def arrange_grid(g, col_order, hue_order, hlines=None, ylims=None, yticks=None):
    for i_col, col in enumerate(col_order):
        for ax in g.axes[:, i_col]:
            if len(col_order) == 1:
                ax.set_xticks(np.arange(len(hue_order)))
                ax.set_xticklabels(hue_order, rotation=45, ha='right', rotation_mode='anchor')
            else:
                ax.set_xticks([(len(hue_order) - 1) / 2])
                ax.set_xticklabels([col])
            if ylims is not None:
                ax.set_ylim(ylims)
                ax.set_yticks(yticks)
            ax.set_xlabel('')
            if i_col > 0:
                ax.spines['left'].set_color('none')
                ax.tick_params(axis='y', length=0)  # hide ticks without affecting leftmost axis
            else:
                ax.spines['left'].set_position(("axes", -0.15))
            if hlines is not None:
                ax.axhline(hlines, ls='--', color=[.5] * 3, lw=1, zorder=0)


def general_stats(pseudo, stat_df, depvar, ax, roi, class_name, class_labels, test, use_keys, df_id):

    print(use_keys)

    # skip_int = 0 if class_name == 'helper' and (not pseudo or not set(use_keys).isdisjoint(['odor', 'ccgp', 'parallel'])) else 1
    skip_int = 0 if len(class_labels) == 1 and (not pseudo or not set(use_keys).isdisjoint(['odor', 'ccgp', 'var', 'parallel'])) else 1
    minus = 0 if pseudo or not set(use_keys).isdisjoint(['odor', 'ccgp', 'var', 'parallel']) else 1
    start = 1 + len(class_labels) - 1 + len(use_keys) - 1 - minus
    end = start + len(use_keys) - 1 - minus

    print(skip_int, minus, start, end)

    if not set(use_keys).isdisjoint(['odor', 'ccgp', 'var', 'parallel']):
        centers = [0] if class_name == 'helper' else [0.5]
    elif df_id == 'pool':
        centers = [0.5]
    else:
        centers = np.arange(skip_int, len(use_keys) - minus)

    # if len(stat_df) == len(np.unique(stat_df['mouse'])):
    #     print('Only one observation per group. Using ANOVA to avoid instability')
    #     stat_out = run_lm('Subregion', roi, class_name, stat_df, depvar=depvar, lme=False, test=test)
    #     test = 'anova'
    # else:
    # I think it's better to keep it this way. In theory, this model should be equivalent to ANOVA when there is only
    # one observation per group, but in practice they differ slightly, perhaps due to the way the models are fit/the
    # way significance is computed in each case. (ANOVA is more conservative in general.)
    stat_out = run_lm('Subregion', roi, class_name, stat_df, depvar=depvar, lme=True, test=test)
    # if len(stat_out) == 0:
    if type(stat_out) != MixedLMResultsWrapper:
        print('LinAlgError detected')
        return stat_out, [1]

    if test == 'lme':
        result = stat_out.summary().tables[1]
        print(stat_out.summary())
        # print(result)
        # if class_name == 'helper':  # skip intercept term, usually
        if len(class_labels) == 1:
            relevant_ps = result[skip_int:len(use_keys) - minus]['P>|z|'].values
        else:
            # interaction terms when they exist (i.e. not for ccgp or odor), else main effect of grouping
            if len(np.unique(stat_df['grouping'])) > 1:
                # relevant_ps = result[-(len(use_keys) - minus):-1]['P>|z|'].values.astype(np.float64)
                relevant_ps = result[start:end]['P>|z|'].values  #.astype(np.float64)
            else:
                relevant_ps = result[-2:-1]['P>|z|'].values #.astype(np.float64)
        # print(relevant_ps)
        relevant_ps[relevant_ps == ''] = 1
        relevant_ps = relevant_ps.astype(np.float64)

    elif test == 'anova':
        print(stat_out)
        if len(class_labels) == 1: relevant_ps = stat_out['PR(>F)'][skip_int:len(use_keys) - minus].values
        elif len(np.unique(stat_df['grouping'])) > 1: relevant_ps = stat_out['PR(>F)'][start:end].values
        else: relevant_ps = stat_out['PR(>F)'][-2:-1].values

    else:
        result = stat_out
        relevant_ps = result[result['Source'] == 'Interaction']['p-unc'].values.astype(np.float64)
        if np.isnan(relevant_ps):
            relevant_ps = result[result['Source'] == 'grouping']['p-unc'].values.astype(np.float64)

    # ax = g.axes.flat[i_roi]
    print(centers, relevant_ps)
    plot_stars(ax, centers, np.array(relevant_ps))

    return stat_out, relevant_ps


def temporal_plotter(sub_df, class_name, class_labels, depvar, use_keys, use_colors, class_colors, rois, per_order,
                     centers, hlines=0.5, ylims=None, yticks=None):

    row_name, row_order, col_name, col_order, hue, hue_order = get_row_col_hue(rois, class_name, class_labels, use_keys)
    palette = use_colors if hue == 'grouping' else class_colors

    grid_kwargs = dict(row=row_name, row_order=row_order, col=col_name, col_order=col_order, height=2, aspect=3/2,
                       sharex=False, gridspec_kws={'wspace': 0.1, 'hspace': 0.5})
    hue_kwargs = dict(x='period', y=depvar, hue=hue, hue_order=hue_order, palette=palette, estimator='mean', errorbar=('ci', 95))
    # take the mean across sessions/pseudopopulations
    use_df = sub_df.groupby(['mouse', 'Subregion', class_name, 'grouping', 'period']).mean().reset_index()
        # .sort_values(by='period', key=lambda series: [per_order.index(x) for x in series]).reset_index()
    time_df = use_df.replace({'period': {k: v for k, v in zip(per_order, centers)}})
    # print(time_df)
    # mouse_avg_df = use_df.groupby(['Subregion', class_name, 'grouping']).agg({depvar: ['mean', 'std', 'sem']})
    g = sns.FacetGrid(data=time_df, **grid_kwargs)
    g.map_dataframe(sns.lineplot, **hue_kwargs)
    # g.set(xticks=per_order[::4], xticklabels=centers[::4])
    g.set(title='', xlabel='')
    g.add_legend(loc=(1.04, .5))

    trace_dict = {'cs_in': 0,
                  'cs_out': 1,
                  'trace_end': 3,
                  'xlim': (-1, 5),
                  'ylabel': 'Decoder\naccuracy',
                  'xlabel': 'Time from CS (s)'
                  }

    for i_col, col in enumerate(col_order):
        for i_row, ax in enumerate(g.axes[:, i_col]):
            setUpLickingTrace(trace_dict, ax, override_ylims=True)
            # ax.set_xticklabels([col])
            # ax.set_xlabel('')
            # ax.set_xticks([0, 2, 4], [0, 2, 4])
            if i_row == 0:
                ax.set_title(col)
            if ylims is not None:
                ax.set_ylim(ylims)
                ax.set_yticks(yticks)
            # if i_row == (len(row_order) - 1):
            #     ax.set_xlabel('Time from CS (s)'
            #     ax.set_xlabel('Time from CS (s)')
            if i_col > 0:
                ax.spines['left'].set_color('none')
                ax.tick_params(axis='y', length=0)  # hide ticks without affecting leftmost axis
            else:
                ax.spines['left'].set_position(("axes", -0.15))
            if hlines is not None:
                ax.axhline(hlines, ls='--', color=[.5] * 3, lw=1, zorder=0)


def plot_avg_confusion_mat_by_class(class_name, class_labels, dec_dict, per_keys, n_trace_types, pi, beh=False,
                                    pseudo=False, activity_type='spks', mouse_colors=None, class_colors=None):

    # plot confusion matrices for odor decoding
    scores = dec_dict['per']['odor']['scores']
    n_rois = 1  # for now, always All Subregions or behavior
    if len(class_labels) > 1:
        ncol = len(class_labels) + 1
        nrow = n_rois
    else:
        ncol = n_rois
        nrow = 1

    across_dist_inds = [[np.array([2, 2, 3, 3]), np.array([4, 5, 4, 5])], [np.array([4, 5, 4, 5]), np.array([2, 2, 3, 3])]]
    within_dist_inds = [[np.array([2, 2, 3, 3]), np.array([2, 3, 2, 3])], [np.array([4, 5, 4, 5]), np.array([4, 4, 5, 5])]]
    use_keys = ['Across distribution', 'Within distribution']
    use_colors = ['#74B72E', '#FF6600']
    save_dir = get_save_dir(False, beh)

    for per_key in per_keys:
        print(per_key)
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2), squeeze=False)
        avg_confs = []

        odor_data = {'grouping': [],
                     'prob': [],
                     'mouse': [],
                     'pop_id': [],
                     class_name: []}
                     # 'per_key': []}

        for i_cls, class_label in enumerate(class_labels):
            # n_rois = len(scores[class_label].keys())
            # fig, axs = plt.subplots(1, n_rois, figsize=(n_rois * 2, 2), squeeze=False)
            # for i_roi, roi in enumerate(scores[class_label].keys()):
            roi = 'behavior' if beh else 'All Subregions'
            mouse_scores = {key: val for key, val in scores[class_label][roi].items() if key != 'all_mice'}
            n_mice = len(mouse_scores)
            confusion_mats = np.zeros((n_mice, n_trace_types, n_trace_types))
            for i_mouse, mouse_name in enumerate(mouse_scores.keys()):
                mouse_data = mouse_scores[mouse_name]
                if pseudo:
                    confs = list(mouse_data['pseudo']['odor'][per_key][3][-1])  # -1 is the max pseudopop size
                else:
                    confs = [mouse_data[fig_path]['odor'][per_key][2] for fig_path in mouse_data.keys() if
                             fig_path != 'pseudo']
                norm_confs = np.array([conf / np.sum(conf, axis=-1, keepdims=True) for conf in confs])
                confusion_mats[i_mouse] = np.mean(norm_confs, axis=0)  # average across sessions within mouse

                n_pops = norm_confs.shape[0]
                across_dist_probs = np.concatenate([norm_confs[:, rows, cols].reshape((n_pops, 4)) for rows, cols in across_dist_inds], axis=1)
                within_dist_probs = np.concatenate([norm_confs[:, rows, cols].reshape((n_pops, 4)) for rows, cols in within_dist_inds], axis=1)
                # print(across_dist_probs.shape, within_dist_probs.shape)
                odor_data['prob'].extend(np.mean(across_dist_probs, axis=1))
                odor_data['prob'].extend(np.mean(within_dist_probs, axis=1))


                odor_data['grouping'].extend(np.repeat(use_keys, n_pops))
                odor_data['mouse'].extend([mouse_name] * n_pops * 2)
                odor_data['pop_id'].extend(np.tile(np.arange(n_pops), 2))
                odor_data[class_name].extend([class_label] * n_pops * 2)
                # odor_data['per_key'].extend([per_key] * n_pops)

            avg_confs.append(confusion_mats.mean(axis=0))
            ax = axs[0, i_cls]
            im = plot_confusion(confusion_mats.mean(axis=0), ax, class_label, normalize=True)  # average across mice

        add_cbar(fig, im, 'Probability of classification')
        # ax.colorbar(im)

        if len(class_labels) > 1:
            odor_df = pd.DataFrame(odor_data)
            pivot_df = odor_df.pivot(columns=class_name, index=['grouping', 'mouse', 'pop_id'],
                                     values='prob').reset_index()
            pivot_df['diff'] = pivot_df[class_labels[1]] - pivot_df[class_labels[0]]
            pivot_df['Subregion'] = 'All Subregions'
            diff_label = f'{class_labels[1]} - {class_labels[0]}'
            pivot_df[class_name] = diff_label
            g, stat_df = general_plotter(pseudo, pivot_df, class_name, [diff_label], 'diff', 'pop_id', use_keys,
                                         use_colors, ['All Subregions'], mouse_colors, class_colors, hlines=0)
            general_stats(pseudo, stat_df, 'diff', g.axes.flat[0], 'All Subregions', class_name, [diff_label], 'lme',
                          use_keys, 'pool')
            plt.savefig(os.path.join(save_dir, 'mean_across_within_class_prob_{}_{}_{}.pdf'.format(
                class_label, per_key, activity_type)), bbox_inches='tight')

            ax = axs[0, -1]
            diff_mat = avg_confs[1] - avg_confs[0]
            cmap = cmocean.tools.crop(cmocean.cm.balance, vmin=np.amin(diff_mat), vmax=np.amax(diff_mat), pivot=0)
            im = ax.imshow(diff_mat, cmap)
            add_cbar(fig, im, class_labels[1] + '$-$' + class_labels[0], left=.97, width=.012)

        for ax in axs.flat:
            ax.set_yticks([])
            ax.set_xticks(np.arange(pi['n_trace_types']))
            ax.set_xticklabels(pi['trace_type_names'], rotation=90)

        axs[0, 0].set_ylabel('True Label')
        axs[0, 0].set_yticks(np.arange(pi['n_trace_types']))
        axs[0, 0].set_yticklabels(pi['trace_type_names'])

        fname = os.path.join(save_dir, 'avg_confusion_mat_across_mice_{}_{}_{}'.format(class_label, per_key, activity_type))
        plt.figure(fig)
        plt.savefig(fname + '.pdf', bbox_inches='tight')
        plt.savefig(fname + '.svg', bbox_inches='tight')
        plt.savefig(fname + '.png', bbox_inches='tight', dpi=300)


def get_save_dir(rem=False, beh=False):
    if beh:
        save_dir = 'behavioral_decoding/'
    # elif rem:
    #     save_dir = 'neural_decoding/dist_removed/'
    else:
        save_dir = 'neural_decoding'  #/all_neurons/'

    return save_dir


def plot_mouse_decode_by_class(class_name, class_labels, dec_dict, per_keys, rois=['All Subregions'], pseudo=True,
                               n_splits=6, rem=False, beh=False, use_pop='max', mouse_colors=None, class_colors=None,
                               reg_C=5e-3, test='lme', ylims_kw={}, activity_type='spks'):

    n_train = n_splits - 1
    # assert len(rois) == 1 or len(class_labels) == 1
    assert mouse_colors is not None

    save_dir = get_save_dir(rem, beh)

    for time_key, time_bin in zip(dec_dict.keys(), dec_dict.values()):
    # for time_key, time_bin in zip(['per'], [dec_dict['per']]):
        for i_grp, grp in enumerate(time_bin.values()):

            # if grp['name'] in ['var']:

            print(grp['name'])
            if 'stats' not in grp:
                grp['stats'] = {}
            if 'all_mice' not in grp['stats']:
                grp['stats']['all_mice'] = {}

            if grp['name'] in ylims_kw:
                ylims = ylims_kw[grp['name']]['ylims']
                yticks = ylims_kw[grp['name']]['yticks']
            else:
                ylims = None
                yticks = None

            print(grp['name'], [grp['colors'], grp['pooled_colors']], [grp['keys'], grp['pooled_keys']])
            for df_id, use_colors, use_keys in zip(['disagg', 'pool'], [grp['colors'], grp['pooled_colors']],
                                                   [grp['keys'], grp['pooled_keys']]):
            # for df_id, use_colors, use_keys in zip(['pool'], [grp['pooled_colors']], [grp['pooled_keys']]):

                if not ((grp['name'] in ['odor'] and df_id == 'pool') or (grp['name'] in ['ccgp'] and df_id == 'disagg')):

                    if pseudo:
                        # take the average over all pseudopopulations
                        # sub_df = grp['pseudo_mouse_dfs'][df_id][per_key].groupby(['mouse', 'grouping', class_name, 'Subregion']).mean().reset_index()
                        # sub_df = grp['pseudo_mouse_dfs'][df_id][per_key][use_pop]
                        prefix = 'pseudo_'
                        strat = 'pop_id'
                        within = False

                    else:
                        # sub_df = grp['mouse_dfs'][df_id][per_key]
                        prefix = ''
                        strat = 'fig_path'
                        within = True

                    if grp['name'] == 'odor':
                        hlines = 1 / grp['resps']['odor'].shape[0]
                    else:
                        hlines = .5

                    # if len(rois) == 1:
                    #     sub_df = sub_df[sub_df['Subregion'] == rois[0]]
                    #     use_subr = rois[0]
                    # else:
                    #     use_subr = 'across_subr'

                    if time_key == 'per':

                        # cross-temporal decoding result
                        n_pers = int(np.sqrt(len(per_keys)))
                        row_name, row_order, col_name, col_order, hue, hue_order = get_row_col_hue(rois, class_name, class_labels, use_keys)
                        # print(row_order, col_order, hue_order)
                        col_order = hue_order if len(row_order) == 1 and len(col_order) == 1 else col_order
                        ctd_mat = np.full((n_pers, n_pers, len(row_order), len(col_order)), np.nan)
                        ctd_stat = np.full((n_pers, n_pers, len(row_order), len(col_order)), np.nan)

                        for i_per, per_key in enumerate(per_keys):

                            print(per_key)

                            sub_df = grp[prefix + 'mouse_dfs'][df_id][per_key]
                            if pseudo: sub_df = sub_df[use_pop]
                            sub_df = sub_df.groupby(['mouse', 'Subregion', class_name, 'grouping', strat, 'period'], as_index=False).mean()
                            # print(sub_df)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                g, stat_df = general_plotter(pseudo, sub_df, class_name, class_labels, 'Accuracy',
                                                             strat, use_keys, use_colors, rois, mouse_colors, class_colors,
                                                             within=within, hlines=hlines, ylims=ylims, yticks=yticks)
                            # print(stat_df)
                            # stat_df is a df with the differences (Across - Within)
                            agg_df = stat_df.groupby(['mouse', 'Subregion', class_name, 'grouping']).mean().reset_index()
                            # print(agg_df)
                            # .sort_values(by=class_name, key=lambda series: [class_labels.index(x) for x in series]).reset_index()
                            diff_df = agg_df[agg_df['grouping'] == use_keys[0]]
                            # print(diff_df)
                            if grp['name'] not in ['odor', 'ccgp', 'var'] and pseudo and df_id == 'pool':
                                diff_df['Accuracy'] -= agg_df.loc[agg_df['grouping'] == use_keys[1], 'Accuracy'].values

                            ml = prefix + 'mouse_level'
                            for i_roi, roi in enumerate(rois):

                                if roi not in grp['stats']['all_mice']:
                                    grp['stats']['all_mice'][roi] = {}
                                if ml not in grp['stats']['all_mice'][roi]:
                                    grp['stats']['all_mice'][roi][ml] = {}
                                if df_id not in grp['stats']['all_mice'][roi][ml]:
                                    grp['stats']['all_mice'][roi][ml][df_id] = {}

                                grp['stats']['all_mice'][roi][ml][df_id][per_key] = general_stats(
                                    pseudo, stat_df, 'Accuracy', g.axes.flat[i_roi], roi, class_name, class_labels, test, use_keys, df_id)

                                if df_id == 'pool' or grp['name'] == 'odor':
                                    roi_mean = diff_df[diff_df['Subregion'] == roi].groupby(class_name).mean().sort_values(
                                        by=class_name, key=lambda series: [class_labels.index(x) for x in series]).reset_index()['Accuracy']
                                    # roi_mean = [agg_df.loc[np.logical_and(
                                    #     agg_df['Subregion'] == roi,agg_df[class_name] == class_label, 'Accuracy'].mean() for class_label in class_labels]
                                    pval = grp['stats']['all_mice'][roi][ml][df_id][per_key][1]

                                    if row_name == class_name:
                                        if len(roi_mean) > 0:
                                            ctd_mat[i_per // n_pers, i_per % n_pers, :, i_roi] = roi_mean
                                            ctd_stat[i_per // n_pers, i_per % n_pers, :, i_roi] = pval
                                        else:
                                            ctd_mat[i_per // n_pers, i_per % n_pers, :, i_roi] = np.nan
                                            ctd_stat[i_per // n_pers, i_per % n_pers, :, i_roi] = 1
                                    else:
                                        ctd_mat[i_per // n_pers, i_per % n_pers, i_roi, :] = roi_mean
                                        ctd_stat[i_per // n_pers, i_per % n_pers, i_roi, :] = pval


                                # use_keys == 1 occurs with odor and ccgp. In general_stats, I've already taken the p-value of the
                                # intercept or the class_name effect (lesion or genotype), so don't need to do it here as well
                                if within and len(use_keys) > 1:  # ~stat_df.equals(sub_df):
                                    pvals = []
                                    for grouping in use_keys:
                                        print(grouping)
                                        trim_df = sub_df[np.logical_and(sub_df['grouping'] == grouping, sub_df['Subregion'] == roi)]
                                        to_fit = trim_df.groupby(
                                            ['mouse', 'fig_path', class_name]).agg({'Accuracy': 'mean'}).reset_index()
                                        # compare it to chance
                                        to_fit['Accuracy'] -= hlines
                                        formula = 'Accuracy ~ 1 + C({})'.format(class_name)
                                        try:
                                            with warnings.catch_warnings():
                                                warnings.simplefilter('ignore')
                                                mfit = mixedlm(formula, to_fit, groups='mouse').fit(method=['powell', 'lbfgs'], maxiter=2000)
                                            print(mfit.summary())
                                            pval = mfit.summary().tables[1][-2:-1]['P>|z|'].values.astype(np.float64)
                                        except ValueError:  # no pvalue because did not converge
                                            pval = 1
                                        except LinAlgError:  # Singular matrix (too few samples)
                                            pval = 1
                                        # print(pval)
                                        pvals.append(pval)
                                    print(pvals)
                                    plot_stars(g.axes.flat[i_roi], np.arange(len(use_keys)), np.array(pvals), ytop_scale=1.1)

                            save_mouse_plot(save_dir, grp, n_train, reg_C, test, df_id, rois, pseudo, per_key, use_pop, class_name, activity_type)

                        if df_id == 'pool' or grp['name'] == 'odor':

                            fig, axs = plt.subplots(len(row_order), len(col_order), figsize=(len(col_order)*2, len(row_order)*2), squeeze=False)
                            try:
                                cmap = cmocean.tools.crop(cmocean.cm.balance, ctd_mat.min(), ctd_mat.max(), pivot=0)
                            except AssertionError:
                                cmap = cmocean.cm.dense
                            for i_row, row in enumerate(row_order):
                                for i_col, col in enumerate(col_order):
                                    ax = axs[i_row, i_col]
                                    if i_row == 0: ax.set_title(col)
                                    if i_row == len(row_order) - 1: ax.set_xlabel('Time from CS (s)')
                                    if i_col == 0: ax.set_ylabel('Time from CS (s)')
                                    im = ax.imshow(ctd_mat[:, :, i_row, i_col], cmap=cmap)
                                    ax.set_xticks(np.arange(-.5, n_pers))
                                    ax.set_xticklabels(np.arange(0, n_pers+1))
                                    ax.set_yticks(np.arange(-.5, n_pers))
                                    ax.set_yticklabels(np.arange(0, n_pers + 1))
                                    for i in range(n_pers):
                                        for j in range(n_pers):
                                            ax.text(j, i, get_stars(ctd_stat[i, j, i_row, i_col]), ha="center",
                                                    va="center", color="w", fontdict={'family': 'DejaVu Sans', 'size': 10})

                            # clab = 'Accuracy: {}$-$\n{}'.format(use_keys[0].split(' ')[0], use_keys[1])
                            clab = 'Accuracy: {}'.format('$-$\n'.join(use_keys))
                            add_cbar(fig, im, clab, pad=40, width=.05 / len(col_order))
                            fig.suptitle(grp['name'])
                            hide_spines()
                            save_mouse_plot(save_dir, grp, n_train, reg_C, test, df_id, rois, pseudo, 'ctd', use_pop, class_name, activity_type)

                    elif pseudo == False:
                        per_df = grp[prefix + 'mouse_dfs']['all'].copy()
                        if pseudo:
                            per_df = per_df[per_df['max']]
                        if df_id == 'pool':
                            per_df = determine_grouping(grp, per_df)

                            # is_within_dist = np.isin(per_df['grouping'], grp['within_dist_keys'])
                            # per_df.loc[is_within_dist, 'grouping'] = grp['pooled_keys'][1]
                            # per_df.loc[~is_within_dist, 'grouping'] = grp['pooled_keys'][0]

                        per_order = ['{}_{}'.format(x, x) for x in np.arange(len(grp['time_bin_centers']))]
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            temporal_plotter(per_df, class_name, class_labels, 'Accuracy', use_keys, use_colors, class_colors, rois,
                                             per_order, grp['time_bin_centers'], hlines=hlines, ylims=ylims, yticks=yticks)
                        save_mouse_plot(save_dir, grp, n_train, reg_C, test, df_id, rois, pseudo, time_key, use_pop, class_name, activity_type)


def save_mouse_plot(save_dir, grp, n_train, reg_C, test, df_id, rois, pseudo, time_label, use_pop, class_name, activity_type):
    fname = save_dir + '_'.join([grp['name'], 'n_train', str(n_train), 'C', str(reg_C), 'test', test, df_id,
                                 '_'.join(rois), 'pseudo', str(pseudo), 'by', class_name, time_label, 'pop',
                                 use_pop, 'across', 'mice', activity_type])
    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png', bbox_inches='tight')


def plot_parallelism_interaction_by_class(class_name, ps_resps, periods, rois=['All Subregions'], rem=False,
                                          mouse_colors=None, class_colors=None, pseudo=False, test='lme', activity_type='spks'):

    assert mouse_colors is not None

    ps_colors = ['#234d20', '#36802d']  #, '#FF6600']
    # pooled_colors = ['#74B72E', '#C21A09']
    pooled_colors = ['k']  # ['#74B72E', '#FF6600']
    # ps_keys = ['B1-C1v.B2-C2', 'B1-C2v.B2-C1', 'B1-B2v.C1-C2']
    ps_keys = ['Variance direction 1', 'Variance direction 2']  # , 'Odor direction']
    pooled_keys = ['parallel'] # , 'Within distribution']

    class_labels = list(ps_resps.keys())
    assert len(class_labels) == 1 or len(rois) == 1

    ps_dict = make_parallelism_interaction_df_by_class(class_name, ps_resps, periods)
    disagg_df = pd.DataFrame(ps_dict)

    pool_df = disagg_df.copy()
    pool_df['grouping'] = pooled_keys[0]

    ps_df = {'disagg': disagg_df, 'pool': pool_df}
    ps_stats = {df_id: {roi: {per: {} for per in periods['periods_to_plot']} for roi in rois} for df_id in ['disagg', 'pool']}

    # save_dir = ''  # 'dist_removed/' if rem else 'all_neurons/'

    for per in periods['periods_to_plot']:

        # for df_id, use_colors, use_keys in zip(['disagg', 'pool'], [ps_colors, pooled_colors], [ps_keys, pooled_keys]):
        # df_id = 'disagg' if class_name == 'genotype' else 'pool'
        df_id = 'pool'
        use_colors = pooled_colors
        use_keys = pooled_keys

        if len(rois) == 1:
            use_subr = rois[0]
            use_df = ps_df[df_id][np.logical_and(ps_df[df_id]['period'] == per, ps_df[df_id]['Subregion'] == use_subr)]
            plot_df = ps_df['pool'][np.logical_and(ps_df['pool']['period'] == per, ps_df['pool']['Subregion'] == use_subr)]
        else:
            use_subr = 'across_subr'
            use_df = ps_df[df_id][ps_df[df_id]['period'] == per]
            plot_df = ps_df['pool'][ps_df['pool']['period'] == per]

        if 'mouse' in use_df.keys() and len(np.unique(use_df['mouse'])) > 1:
            strat = 'fig_path'
            unit = 'mouse'
        else:
            strat = 'pop_id'
            unit = 'pop_id'
        # print(use_df.keys(), strat, unit)

        # average over the two ways to draw parallelism score
        g, stat_df = general_plotter(pseudo, plot_df, class_name, class_labels, 'Parallelism', strat,
                                     use_keys, use_colors, rois, mouse_colors, class_colors, within=False, hlines=0, unit=unit)

        # grouping_ps_all = []
        for i_roi, roi in enumerate(rois):

            cls_ps = []
            print(per, roi)
            ps_stats[df_id][roi][per] = general_stats(pseudo, stat_df, 'Parallelism', g.axes.flat[i_roi], roi,
                                                      class_name, class_labels, test, use_keys, df_id)

            if class_name == 'genotype':
                for class_label in class_labels:
                    print(class_label)
                    cls_df = stat_df[stat_df[class_name] == class_label]
                    # print(cls_df)
                    print(per, roi, class_label)
                    # use this test for imaging analysis when I combine across mice ONLY
                    stat_out = run_lm('Subregion', roi, class_name, cls_df, depvar='Parallelism', lme=True, test='lme', groups='pop_id')
                    try:
                        print(stat_out.summary())
                        cls_p = stat_out.summary().tables[1]['P>|z|'][0]
                        cls_p = float(cls_p)
                    except:
                        cls_p = 1
                    cls_ps.append(cls_p)

                print(cls_ps)
                plot_stars(g.axes.flat[i_roi], np.arange(len(class_labels)), cls_ps, ytop_scale=.95)

        title = '_'.join([periods['period_names'][per], use_subr])

        plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
            [class_name, 'per', title, df_id, 'pseudo', str(pseudo), df_id, activity_type, 'interaction', 'parallelism_score']) + '.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
            [class_name, 'per', title, df_id, 'pseudo', str(pseudo), df_id, activity_type, 'interaction', 'parallelism_score']) + '.png'), bbox_inches='tight')

    return ps_stats


def make_ps_df_by_class(class_name, use_resps, periods):
    use_dict = {  # 'mouse': [],
        'grouping': [],
        'period': [],
        class_name: [],
        'Subregion': [],
        'Parallelism': []}

    #     for i_mouse, mouse_name in enumerate(mice):
    #         if mouse_name in grp['all']:
    for class_label in use_resps.keys():
        for roi in use_resps[class_label].keys():
            for key in use_resps[class_label][roi].keys():
                n_shuff = np.shape(use_resps[class_label][roi][key][0])[0]
                use_dict['grouping'].extend([key] * periods['n_prerew_periods'] * n_shuff)
                use_dict[class_name].extend([class_label] * periods['n_prerew_periods'] * n_shuff)
                use_dict['Subregion'].extend([roi] * periods['n_prerew_periods'] * n_shuff)
                for per in range(periods['n_prerew_periods']):
                    use_dict['period'].extend([per] * n_shuff)
                    use_dict['Parallelism'].extend(use_resps[class_label][roi][key][per])

    return use_dict


def plot_parallelism_score_by_class(class_name, ps_resps, ps_shuff, periods, rois=['All Subregions']):

    ps_colors = ['#234d20', '#36802d', '#ff0000']

    class_labels = ps_resps.keys()
    assert len(class_labels) == 1 or len(rois) == 1

    ps_dict = make_ps_df_by_class(class_name, ps_resps, periods)
    shuff_dict = make_ps_df_by_class(class_name, ps_shuff, periods)

    ps_df = pd.DataFrame(ps_dict)
    shuff_ps_df = pd.DataFrame(shuff_dict)

    if len(rois) > 1:
        spec = dict(x='Subregion', y='Parallelism', hue='grouping')
        n_labels = len(rois)  # number of subregions
        ps_ps = np.zeros((1, len(rois), 3))
    else:
        spec = dict(x=class_name, y='Parallelism', hue='grouping')
        n_labels = len(ps_resps.keys())  # number of genotypes, lesion, etc.
        ps_ps = np.zeros((n_labels, 1, 3))

    for per in periods['periods_to_plot']:

        print(periods['period_names'][per])

        if len(rois) == 1:
            use_subr = rois[0]
            use_df = ps_df[np.logical_and(ps_df['period'] == per, ps_df['Subregion'] == use_subr)]
            use_shuff = shuff_ps_df[np.logical_and(shuff_ps_df['period'] == per, shuff_ps_df['Subregion'] == use_subr)]
        else:
            use_subr = 'across_subr'
            use_df = ps_df[ps_df['period'] == per]
            use_shuff = shuff_ps_df[shuff_ps_df['period'] == per]

        plt.figure(figsize=(n_labels+1, 2))
        g = sns.stripplot(**spec, data=use_df, palette=ps_colors, dodge=True, size=8)
        sns.violinplot(**spec, data=use_shuff, color=[.8, .8, .8])

        # don't include the black dots in the legend
        h, l = g.get_legend_handles_labels()
        nlab = len(np.unique(use_df['grouping']))
        plt.legend(h[nlab:], l[nlab:], loc=(1.04, 0))

        for i_label, class_label in enumerate(ps_shuff.keys()):
            for i_roi, roi in enumerate(rois):
                for i_key, key in enumerate(ps_shuff[class_label][roi].keys()):
                    ps_ps[i_label, i_roi, i_key] = 2 * (1 - np.mean(
                        ps_resps[class_label][roi][key][per][0] > ps_shuff[class_label][roi][key][per]))

        xpos = (np.array([[-.28, 0, .28]]) + np.arange(n_labels)[:, np.newaxis]).flatten()

        title = '_'.join([periods['period_names'][per], use_subr])
        # plt.title(title, pad=20)

        # plt.xticks(np.arange(periods['n_periods']), periods['period_names'], rotation=45)
        plt.xlim(-0.5, n_labels - 0.5)
        plt.vlines(np.arange(0.5, plt.xlim()[1]-.5), ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='--', color=[.5] * 3, lw=1)
        plot_stars(plt.gca(), xpos, ps_ps.flatten(), s=12)

        hide_spines()
        plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
            [class_name, 'per', title, 'parallelism_score']) + '.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join('..', 'neural-plots', '_'.join(
            [class_name, 'per', title, 'parallelism_score']) + '.png'), bbox_inches='tight')



def compute_parallelism_score_by_class(class_name, class_labels, neuron_info, dat, shuff_dat, periods,
                                       rois=['All Subregions'], metric='cosine'):
    """
    :param class_name e.g. 'genotype', 'lesion'
    :param class_labels e.g. ['matched', 'lesioned']
    :param neuron_info: pandas DataFrame
    :param dat: trial-averaged data, usually X_means. Shape = (n_trial_types, n_neurons, n_periods)
    :param shuff_dat: trial-averaged data after randomly shuffling trial types, usually X_shuff_means,
    Shape = (n_shuffles, n_trial_types, n_neurons, n_periods)
    :return:
    """

    # parallelism score setup
    ps_keys = ['B1-C1v.B2-C2', 'B1-C2v.B2-C1', 'B1-B2v.C1-C2']

    # types_to_pair = np.array([2, 3, 4, 5])
    n_shuff = shuff_dat.shape[0]
    parallel_stats = np.zeros((3, periods['n_prerew_periods'], len(class_labels), len(rois)))

    pair_order = [(2, 4, 3, 5), (2, 5, 3, 4), (2, 3, 4, 5)]

    ps_resps = {class_label: {roi: {k: {} for k in ps_keys} for roi in rois} for class_label in class_labels}
    ps_shuff = {class_label: {roi: {k: {per: np.zeros(n_shuff) for per in range(periods['n_prerew_periods'])}
                                    for k in ps_keys} for roi in rois} for class_label in class_labels}

    for i_label, class_label in enumerate(class_labels):
        for i_roi, roi in enumerate(rois):
            if class_label != 'all':
                inds = neuron_info[class_name] == class_label
            else:
                inds = np.ones(len(neuron_info), dtype=bool)
            if roi != 'All Subregions':
                inds = np.logical_and(inds, neuron_info['str_regions'] == roi)
            use_dict = ps_resps[class_label][roi]
            use_shuff = ps_shuff[class_label][roi]

            for per in range(periods['n_prerew_periods']):

                for i_po, po in enumerate(pair_order):
                    use_dict[ps_keys[i_po]][per] = [1 - pairwise_distances(
                        np.vstack((dat[po[0], inds, per] - dat[po[1], inds, per],
                                   dat[po[2], inds, per] - dat[po[3], inds, per])),
                        metric=metric)[0, 1]]

                for i_null in range(n_shuff):
                    for i_po, po in enumerate(pair_order):
                        use_shuff[ps_keys[i_po]][per][i_null] = 1 - pairwise_distances(
                            np.vstack((shuff_dat[i_null, po[0], inds, per] - shuff_dat[i_null, po[1], inds, per],
                                       shuff_dat[i_null, po[2], inds, per] - shuff_dat[i_null, po[3], inds, per])),
                            metric=metric)[0, 1]

                parallel_stats[:, per, i_label, i_roi] = [np.mean(use_dict[ps_keys[x]][per] > use_shuff[ps_keys[x]][per]) for x in range(3)]

    return ps_resps, ps_shuff, parallel_stats


def async_sizefun(X, seed, n_shuff=100):
    y = np.tile(A=np.reshape([1, 2], newshape=(2, 1)), reps=[1, X.shape[1]])
    rng = np.random.default_rng(seed=seed)
    n_out = 3
    scores = np.zeros((n_shuff + 1, n_out))
    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    X_train = X[..., 0]
    y_train = y[~np.isnan(X_train)]
    X_train = X_train[~np.isnan(X_train)].reshape(-1, 1)
    X_test = X[..., 1]
    y_test = y[~np.isnan(X_test)]
    X_test = X_test[~np.isnan(X_test)].reshape(-1, 1)

    logregcv = LogisticRegressionCV(cv=kfold, solver='liblinear', n_jobs=1, class_weight='balanced',
                                    max_iter=10000, random_state=1)
    pipeline = Pipeline([('estimator', logregcv)]).fit(X_train, y_train)

    for i_shuff in range(n_shuff + 1):

        if i_shuff > 0:
            # y_train = rng.permutation(y_train)  # TODO: see if this better captures my intuitions
            y_test = rng.permutation(y_test)

        preds = pipeline.predict(X_test)
        scores[i_shuff, 0] = balanced_accuracy_score(y_test, preds)
        predictions = pipeline.predict_proba(X_test)
        scores[i_shuff, 1] = roc_auc_score(y_test, predictions[:, 1], average='macro')
        scores[i_shuff, 2] = pipeline['estimator'].coef_[0, 0]

    return scores


def plot_prctile(agg_df, ax, avg_spec, mouse_spec):
    sns.pointplot(data=agg_df, ax=ax, **avg_spec)
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100)
    sns.lineplot(data=agg_df, ax=ax, **mouse_spec)
    ax.set_xlim(-.4, 1.4)
    return ax


def plot_dist_neurons(neuron_info, dist_neurons, timecourses, times, trace_dict, colors, n_trace_types, max_plot=200,
                      plot_title=False):

    n_cols = 8
    n_rows = min(int(np.ceil(dist_neurons.sum() / n_cols)), max_plot)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), sharex=True)
    use_inds = np.flatnonzero(dist_neurons)

    for i in range(n_rows * n_cols):
        ax = axs.flat[i]
        if i < dist_neurons.sum():
            ax.set_prop_cycle(color=colors['colors'][:n_trace_types])
            ax.plot(times, timecourses[:n_trace_types, use_inds[i], :].T)
            setUpLickingTrace(trace_dict, ax=ax, override_ylims=True)
            ax.set_ylabel('')
            ax.set_xlabel('')
            if plot_title:  # iloc for row and loc for column b/c it's potentially a slice of neuron_info
                ax.set_title(' '.join(neuron_info.iloc[use_inds[i]].loc[['names', 'file_dates', 'neuron_idx_inc']].astype('U')))
        else:
            ax.remove()
    hide_spines()
    fig.tight_layout()


def extend_dicts(this_dict, dec_key, dec_type, class_name, class_label, roi, key, per_key, pc_angles=None,
                 value_angles=None, var_angles=None, value_pc1_angle=None, var_pc2_angle=None,
                 sess_value_pc1_angle=None, sess_var_pc2_angle=None, mouse='all_mice', fig_path='pseudo'):

    if pc_angles is not None:
        n_components = len(pc_angles)
        if type(pc_angles[0]) != list:
            pc_angles = [[x] for x in pc_angles]
            value_angles = [value_angles]
            var_angles = [var_angles]

        n_angles = len(value_angles)
        this_dict['value_angle'].extend(value_angles)
        this_dict['var_angle'].extend(var_angles)
        for i_comp in range(n_components):
            this_dict['pc{}_angle'.format(i_comp + 1)].extend(pc_angles[i_comp])
    else:
        this_dict['value_pc1_angle'].append(value_pc1_angle)
        this_dict['var_pc2_angle'].append(var_pc2_angle)
        this_dict['sess_value_pc1_angle'].append(sess_value_pc1_angle)
        this_dict['sess_var_pc2_angle'].append(sess_var_pc2_angle)
        n_angles = 1

    this_dict['dec_key'].extend([dec_key] * n_angles)
    this_dict['dec_type'].extend([dec_type] * n_angles)
    this_dict[class_name].extend([class_label] * n_angles)
    this_dict['Subregion'].extend([roi] * n_angles)
    this_dict['key'].extend([key] * n_angles)
    this_dict['per_key'].extend([per_key] * n_angles)
    this_dict['i_fold'].extend(np.arange(n_angles))
    this_dict['mouse'].extend([mouse] * n_angles)
    this_dict['fig_path'].extend([fig_path] * n_angles)

    return this_dict


def compute_angles(dec_dict, neuron_info, cue_resps, protocol_info, components, session_components, ret_df, class_name,
                   class_labels, rois, n_splits=6, train_per=3, test_per=3, use_pseudo=True, do_zscore=True):

    """
    :param dec_dict: decoding dictionary
    :param neuron_info: pandas DataFrame
    :param cue_resps: array of shape n_trace_types x n_cells x max_n_trial_types x n_prewrew_periods
    :param protocol_info: dictionary
    :param components: output from sklearn PCA, shape n_components x n_cells
    :param class_name: e.g. 'lesion'
    :param class_labels: e.g. ['lesioned', 'control']
    :param rois: e.g. ['lAcbSh', 'VLS', 'All Subregions']
    :param n_splits: int, e.g. 6
    :train_per: which training period to use
    :test_per: which testing period to use. This should be irrelevant?
    :all_mice: whether to also compute angles for pseudopopulations that are pooled between mice
    :return a DataFrame containing identifying information and the angles formed between the decoder axis and both the
    first PC and the value regression weights
    """

    n_components = components[list(class_labels)[0]].shape[0]

    base_dict = {'dec_key': [],
                  'dec_type': [],
                  class_name: [],
                  'Subregion': [],
                  'key': [],
                  'per_key': [],
                  'i_fold': [],
                  'mouse': [],
                  'fig_path': []
                  }

    angle_dict = copy.deepcopy(base_dict)
    for i_comp in range(n_components):
        angle_dict['pc{}_angle'.format(i_comp+1)] = []
    angle_dict['value_angle'] = []
    angle_dict['var_angle'] = []

    control_dict = copy.deepcopy(base_dict)
    control_dict['value_pc1_angle'] = []
    control_dict['var_pc2_angle'] = []
    control_dict['sess_value_pc1_angle'] = []
    control_dict['sess_var_pc2_angle'] = []

    n_train = n_splits - 1
    per_keys = ['{}_{}'.format(train_per, test_per)]
    all_sids = get_sids(neuron_info)
    means = protocol_info['mean'][:protocol_info['n_trace_types']]
    vars = protocol_info['var'][:protocol_info['n_trace_types']]

    ridge = RidgeCV()
    scaler = StandardScaler()

    if do_zscore:
        pipeline = Pipeline([('transformer', scaler), ('estimator', ridge)])
    else:
        pipeline = Pipeline([('estimator', ridge)])

    for dec_key in ['ccgp', 'pair', 'cong', 'mean']:  # ignore 'odor' here
        grp = dec_dict['per'][dec_key]
        for class_label in class_labels:
            class_inds = neuron_info[class_name] == class_label
            for roi in rois:
                if roi == 'All Subregions':
                    roi_inds = np.ones(len(neuron_info), dtype=bool)
                else:
                    roi_inds = neuron_info['str_regions'] == roi
                mice = list(grp['scores'][class_label][roi].keys())
                try:
                    mice.remove('all_mice')
                except:
                    pass
                for key in grp['keys']:
                    for per_key in per_keys:

                        for mouse_name in mice:
                            mouse_inds = neuron_info['names'] == mouse_name
                            # print(class_label, roi, mouse_name, grp['scores'][class_label][roi][mouse_name].keys())
                            fig_paths = np.unique(ret_df.loc[ret_df['name'] == mouse_name, 'figure_path']) if \
                                np.all(neuron_info['str_regions'] == 'behavior') else list(grp['scores'][class_label][roi][mouse_name].keys())
                            # fig_paths = np.unique(ret_df.loc[ret_df['name'] == mouse_name, 'figure_path'])

                            if use_pseudo:
                                if key in grp['scores'][class_label][roi][mouse_name]['pseudo']:
                                    mouse_coefs = grp['scores'][class_label][roi][mouse_name]['pseudo'][key][per_key][1]
                                    class_reg_mouse_neuron_inds = np.array(grp['scores'][class_label][roi][mouse_name]['pseudo'][key][per_key][2][-1])
                                    selected_comps = [components[class_label][i_comp][np.logical_and(roi_inds, mouse_inds)[class_inds]][class_reg_mouse_neuron_inds] for i_comp in range(n_components)]
                                    mouse_coef_angles = [[angle_between(np.squeeze(mouse_coefs[-1, i_fold, :, :]), x)
                                                         for i_fold in range(n_train)] for x in selected_comps]

                                    data = cue_resps[:, np.logical_and.reduce([class_inds, roi_inds, mouse_inds])][:, class_reg_mouse_neuron_inds]
                                    sids = all_sids[np.logical_and.reduce([class_inds, roi_inds, mouse_inds])][class_reg_mouse_neuron_inds]

                                    mouse_value_coefs = disjoint_regress(data, sids, means, train_per=train_per, test_per=test_per, do_cv=False)
                                    # -1 is here because that corresponds to max_pop
                                    mouse_value_angles = [angle_between(np.squeeze(mouse_coefs[-1, i_fold, :, :]),
                                                                        mouse_value_coefs) for i_fold in range(n_train)]
                                    mouse_var_coefs = disjoint_regress(data, sids, vars, train_per=train_per, test_per=test_per, do_cv=False)
                                    mouse_var_angles = [angle_between(np.squeeze(mouse_coefs[-1, i_fold, :, :]),
                                                                      mouse_var_coefs) for i_fold in range(n_train)]

                                    angle_dict = extend_dicts(angle_dict, dec_key, 'pseudo_mouse', class_name, class_label,
                                                              roi, key, per_key, mouse_coef_angles, mouse_value_angles,
                                                              mouse_var_angles, mouse=mouse_name)

                                    mouse_value_pc1_angle = angle_between(components[class_label][0, np.logical_and(
                                        roi_inds, mouse_inds)[class_inds]][class_reg_mouse_neuron_inds], mouse_value_coefs)
                                    mouse_var_pc2_angle = angle_between(components[class_label][1, np.logical_and(
                                        roi_inds, mouse_inds)[class_inds]][class_reg_mouse_neuron_inds], mouse_var_coefs)

                                    control_dict = extend_dicts(control_dict, dec_key, 'pseudo_mouse', class_name, class_label,
                                                                roi, key, per_key, value_pc1_angle=mouse_value_pc1_angle,
                                                                var_pc2_angle=mouse_var_pc2_angle, mouse=mouse_name)
                                fig_paths.remove('pseudo')

                            for fig_path in fig_paths:

                                fig_path_inds = neuron_info['fig_paths'] == fig_path

                                # perform regression on simultaneously recorded data
                                simul_reg_data = cue_resps[:, np.logical_and.reduce([class_inds, roi_inds, fig_path_inds])][..., train_per]

                                n_notnan = np.sum(~np.isnan(simul_reg_data), axis=2)
                                # print(n_notnan.shape)
                                assert np.all([n_notnan[:, 0] == n_notnan[:, i_cell] for i_cell in range(simul_reg_data.shape[1])])
                                trial_means = np.repeat(means, n_notnan[:, 0])
                                trial_vars = np.repeat(vars, n_notnan[:, 0])

                                # reshape into n_cells x n_trials (of all types)
                                simul_reg_data = np.transpose(simul_reg_data, (1, 0, 2)).reshape((simul_reg_data.shape[1], -1))

                                value_reg = clone(pipeline)
                                value_reg.fit(simul_reg_data[:, ~np.isnan(simul_reg_data[0])].T, trial_means)

                                var_reg = clone(pipeline)
                                var_reg.fit(simul_reg_data[:, ~np.isnan(simul_reg_data[0])].T, trial_vars)

                                simul_coefs = grp['scores'][class_label][roi][mouse_name][fig_path][key][per_key][1]
                                if dec_key == 'ccgp':
                                    simul_coef_angles = [angle_between(np.squeeze(simul_coefs), components[class_label][
                                        i_comp][np.logical_and(fig_path_inds, roi_inds)[class_inds]]) for i_comp in range(n_components)]
                                    simul_value_angles = angle_between(np.squeeze(simul_coefs), value_reg.named_steps['estimator'].coef_)
                                    simul_var_angles = angle_between(np.squeeze(simul_coefs), var_reg.named_steps['estimator'].coef_)

                                else:
                                    simul_coef_angles = [np.mean([
                                        angle_between(np.squeeze(simul_coefs[i_fold, :, :]), components[class_label][i_comp][
                                            np.logical_and(fig_path_inds, roi_inds)[class_inds]]) for i_fold in range(n_train)]) for i_comp in range(n_components)]
                                    simul_value_angles = np.mean([angle_between(np.squeeze(simul_coefs[i_fold, :, :]),
                                                                              value_reg.named_steps['estimator'].coef_) for i_fold in range(n_train)])
                                    simul_var_angles = np.mean([angle_between(np.squeeze(simul_coefs[i_fold, :, :]),
                                                                                var_reg.named_steps['estimator'].coef_) for i_fold in range(n_train)])

                                angle_dict = extend_dicts(angle_dict, dec_key, 'simul', class_name, class_label, roi,
                                                          key, per_key, simul_coef_angles, simul_value_angles, simul_var_angles,
                                                          mouse=mouse_name, fig_path=fig_path)

                                simul_value_pc1_angle = angle_between(components[class_label][0, np.logical_and(
                                    fig_path_inds, roi_inds)[class_inds]], value_reg.named_steps['estimator'].coef_)
                                simul_var_pc2_angle = angle_between(components[class_label][1, np.logical_and(
                                    fig_path_inds, roi_inds)[class_inds]], var_reg.named_steps['estimator'].coef_)

                                control_dict = extend_dicts(control_dict, dec_key, 'simul', class_name, class_label,
                                                            roi, key, per_key, value_pc1_angle=simul_value_pc1_angle,
                                                            var_pc2_angle=simul_var_pc2_angle, mouse=mouse_name, fig_path=fig_path)

                                if roi == 'All Subregions' or roi == 'behavior':
                                    # warning: haven't yet considered on multiple classes
                                    index = np.flatnonzero(ret_df['figure_path'] == fig_path)[0]
                                    session_value_pc1_angle = angle_between(session_components[index][0, :],
                                                                            value_reg.named_steps['estimator'].coef_)
                                    session_var_pc2_angle = angle_between(session_components[index][1, :],
                                                                          var_reg.named_steps['estimator'].coef_)
                                else:
                                    session_value_pc1_angle = np.nan
                                    session_var_pc2_angle = np.nan

                                control_dict = extend_dicts(control_dict, dec_key, 'simul', class_name, class_label,
                                                            roi, key, per_key, value_pc1_angle=simul_value_pc1_angle,
                                                            var_pc2_angle=simul_var_pc2_angle, sess_value_pc1_angle=session_value_pc1_angle,
                                                            sess_var_pc2_angle=session_var_pc2_angle, mouse=mouse_name,
                                                            fig_path=fig_path)


    angle_dict = add_grouping(angle_dict, grp['within_dist_keys'])
    control_dict = add_grouping(control_dict, grp['within_dist_keys'])

    return angle_dict, control_dict


def add_grouping(this_dict, within_dist_keys):
    this_dict = pd.DataFrame(this_dict)

    this_dict['grouping'] = None
    is_within_dist = np.isin(this_dict['key'], within_dist_keys)
    this_dict.loc[is_within_dist, 'grouping'] = 'Within distribution'
    this_dict.loc[~is_within_dist, 'grouping'] = 'Across distribution'

    return this_dict


def determine_grouping(grp, use_grp):
    if grp['name'] in ['odor', 'ccgp', 'var']:
        use_grp['grouping'] = grp['pooled_keys'][0]
    else:
        is_within_dist = np.isin(use_grp['grouping'], grp['within_dist_keys'])
        use_grp.loc[is_within_dist, 'grouping'] = grp['pooled_keys'][1]
        use_grp.loc[~is_within_dist, 'grouping'] = grp['pooled_keys'][0]
    return use_grp


def plot_decoder_angles(dec_dict, control_df, angle_df, class_name, class_labels, rois, n_components=3,
                        mouse_colors=None, dec_types=['pseudo_mouse', 'simul'], savedir='neural_decoding'):

    origvars = ['pc{}_angle'.format(i_comp + 1) for i_comp in range(n_components)] + \
               ['value_angle', 'var_angle', 'value_pc1_angle', 'var_pc2_angle', 'sess_value_pc1_angle', 'sess_var_pc2_angle']
    # longnames = ['decoder\naxis and PC{}'.format(i_comp + 1) for i_comp in range(n_components)] + \
    #             ['decoder\naxis and {} Axis'.format(x) for x in ['Value', 'Variance']] + \
    #             ['PC1 and\nValue Axis', 'PC2 and\nVariance Axis', 'Session PC1 and\nValue Axis', 'Session PC2 and\nVariance Axis']
    longnames = ['Decoder axis and PC{}'.format(i_comp + 1) for i_comp in range(n_components)] + \
                ['Decoder axis and {} axis'.format(x) for x in ['Value', 'Variance']] + \
                ['PC1 and Value axis', 'PC2 and Variance axis', 'Session PC1 and Value axis', 'Session PC2 and Variance axis']

    control_vars = ['value_pc1_angle', 'var_pc2_angle', 'sess_value_pc1_angle', 'sess_var_pc2_angle']

    for origvar, longname in zip(origvars, longnames):
    # for origvar, longname in zip(['sess_value_pc1_angle'], ['Session PC1 and\nValue Axis']):
    #     depvar = 'ortho_dev_deg_' + origvar
        depvar = 'cosine_similarity_' + origvar

        use_df = control_df if origvar in control_vars else angle_df
        # use_df[depvar] = np.degrees(use_df[origvar]) - 90  # deviation from orthogonal
        use_df[depvar] = 1 - use_df[origvar]
        use_df['abs_' + depvar] = np.abs(use_df[depvar])

        # for dec_type in ['all_mice', 'pseudo_mouse', 'simul']:
        for dec_type in dec_types:

            if 'sess' in origvar and dec_type == 'pseudo_mouse': continue

            # unit = 'mouse' if dec_type != 'all_mice' else 'i_fold'

            for i_test in range(2):

                if i_test == 0:  # as before, looking at across vs. within distribution
                    sub_df = use_df[np.logical_and(use_df['dec_type'] == dec_type, use_df['dec_key'] != 'mean')]
                    use_keys = dec_dict['cong']['pooled_keys']
                    suffix = 'dist'
                    colors = dec_dict['cong']['pooled_colors']
                    grp = 'grouping'
                    style_order = list(dec_dict.keys())[:3]

                else:  # compare mean decoding
                    sub_df = use_df[np.logical_and(use_df['dec_type'] == dec_type, use_df['dec_key'] == 'mean')]
                    use_keys = dec_dict['mean']['keys']
                    suffix = 'mean'
                    colors = dec_dict['mean']['colors']
                    grp = 'key'
                    style_order = ['mean']

                if dec_type != 'all_mice':

                    if origvar in control_vars and i_test == 0:

                        # these ones are redundant, so I just need to choose any one of the 18 keys between pair, mean, ccgp, and cong
                        sub_df = sub_df[sub_df['key'] == 'Congruent']
                        # print(sub_df)
                        use_keys = ['Across distribution']  # this is a meaningless dummy variable in this case
                        colors = ['k']
                        style_order = ['cong']
                        # print(sub_df)

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        g2, _ = general_plotter(False, sub_df, class_name, class_labels, depvar, None, use_keys,
                                               colors, rois, mouse_colors, ['k'], within=False, hlines=0,
                                               style_order=style_order, grping=grp)
                    # g.add_legend(loc=(1.04, .5))

                    for i_roi, roi in enumerate(rois):
                        # for i_cls, class_label in enumerate(class_labels):
                        for i_key, dec_key in enumerate(style_order):

                            roi_df = sub_df[np.logical_and(sub_df['Subregion'] == roi, sub_df['dec_key'] == dec_key)]
                            roi_fit = roi_df.groupby(
                                    ['mouse', 'fig_path', 'i_fold', class_name, grp]).agg({'abs_' + depvar: 'mean'}).reset_index()
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                mfit = mixedlm('abs_{} ~ 1 + C({})'.format(depvar, grp), roi_fit, groups='mouse').fit(
                                    method=['powell', 'lbfgs'],  maxiter=2000)
                            print(origvar, dec_type, roi, dec_key)
                            print(mfit.summary())
                            tmp = mfit.summary().tables[1]['P>|z|'][1]
                            print(tmp)
                            grpp = 1 if tmp == '' else float(tmp)

                            grp_keys = ['Across distribution'] if dec_key == 'ccgp' else use_keys
                            for i_grp, grouping in enumerate(grp_keys):
                                trim_df = sub_df[np.logical_and.reduce([sub_df[grp] == grouping, sub_df['Subregion'] == roi,
                                                                        sub_df['dec_key'] == dec_key])]
                                to_fit = trim_df.groupby(
                                    ['mouse', 'fig_path', 'i_fold', class_name]).agg({depvar: 'mean'}).reset_index()
                                # print(to_fit)
                                formula = depvar + ' ~ 1 + C({})'.format(class_name)
                                try:  # this could fail if there is only 1 lowest-level item per mouse, or only one surviving mouse, or even sometimes with two mice
                                    with warnings.catch_warnings():
                                        warnings.simplefilter('ignore')
                                        mfit = mixedlm(formula, to_fit, groups='mouse').fit(method=['powell', 'lbfgs'], maxiter=2000)
                                    print(origvar, dec_type, roi, grouping)
                                    print(mfit.summary())
                                    pval = mfit.summary().tables[1][-2:-1]['P>|z|'].values.astype(np.float64)
                                except ValueError:
                                    # _, pval = stats.ttest_1samp(trim_df.groupby('mouse').agg({depvar: 'mean'})[depvar], popmean=0)
                                    # pval = [pval]
                                    print('ANOVA!')
                                    anova_df = trim_df.groupby(['mouse', class_name]).agg({depvar: 'mean'}).reset_index()
                                    if class_name == 'helper':
                                        model = ols(depvar + ' ~ 1', anova_df).fit()
                                    else:
                                        model = ols(formula, anova_df).fit()
                                    mfit = anova_lm(model, typ=3, robust='hc3')
                                    print(mfit)
                                    # print(mfit)
                                    pval = mfit[-2:-1]['PR(>F)'].values.astype(np.float64)
                                # print(pval)
                                if len(rois) == 1:
                                    if type(g2) != list:
                                        g2 = [g2]
                                    [plot_stars(g.axes[i_roi, i_key], [i_grp], pval) for g in g2]
                                else:
                                    raise NotImplementedError('Expected dec_keys to be the columns. Only run on one roi at a time')
                                    # plot_stars(g.axes[-1, i_roi], [i_grp], pval)
                            [plot_stars(g.axes[i_roi, i_key], [0.5], [grpp], ytop_scale=.9, show_ns=True) for g in g2]

                for ig, g in enumerate(g2):
                    # g.set_ylabels('Angle between {}: Deviation\nfrom orthogonal ($\degree$)'.format(longname))
                    g.set_ylabels('Cosine similarity:\n{}'.format(longname))
                    g.fig.suptitle(dec_type, y=1.05)
                    g.savefig('{}/{}_{}_{}_{}_{}.pdf'.format(savedir, depvar, dec_type, class_name, suffix, str(ig), '_'.join(rois)), bbox_inches='tight')
                    g.savefig('{}/{}_{}_{}_{}_{}.png'.format(savedir, depvar, dec_type, class_name, suffix, str(ig), '_'.join(rois)), bbox_inches='tight', dpi=300)


def compute_model_rsa(ret_df, code_order, all_dists, all_rda, pairwise_dists, rda, mousewise_dists, mousewise_rda, mouse_colors, protocol):
    """
    :param ret_df: DataFrame of length n_sessions, with keys 'name' and 'figure_path', in the same order as `pairwise_dists` and `rda`
    :param code_order: list of codes to compare
    :param all_dists: From generate_model_predictions. (n_subsets, n_models, n_components, n_trace_types, n_trace_types).
    It's the Euclidean distances between all trial types, measured in PC space. subsets is usually ['all'] or ['all', 'pess', 'opt']
    :param all_rda: From generate_model_predictions. (n_subsets, n_models, n_trace_types, n_trace_types). It's the cosine
    distances measured in firing rate space, so no components needed.
    :param pairwise_dists: (n_sessions, n_components, n_trace_types, n_trace_types)
    :param rda: (n_sessions, n_trace_types, n_trace_types)
    :param mousewise_dists: (n_mice, n_class_labels, n_components, n_trace_types, n_trace_types)
    :param mousewise_rda: (n_mice, n_class_labels, 1, n_trace_types, n_trace_types)
    :param mouse_colors: dict mapping mouse_name -> color
    :param protocol:
    :return:
    """

    model_palette = sns.husl_palette(len(code_order), l=.5, h=1)
    n_codes = len(code_order)

    rsa_dict = {'name': [], 'figure_path': [], 'code': [], 'i_code': [], 'r': [], 'metric': []}
    rsa_mouse_dict = {'name': [], 'code': [], 'i_code': [], 'r': [], 'metric': []}
    metrics = ['pc1', 'pc2', 'cosine']

    for i_mouse, mouse_name in enumerate(np.unique(ret_df['name'])):
        rsa_mouse_dict['name'].extend([mouse_name] * n_codes * 3)
        rsa_mouse_dict['code'].extend(code_order * 3)
        rsa_mouse_dict['i_code'].extend(list(range(n_codes)) * 3)
        rsa_mouse_dict['metric'].extend(np.repeat(metrics, n_codes))
        for metric, modelmat, datamat in zip(metrics,
                                             [all_dists[0, :, 0], all_dists[0, :, 1], all_rda[0, :]],
                                             [mousewise_dists[i_mouse, 0, 0], mousewise_dists[i_mouse, 0, 1], mousewise_rda[i_mouse, 0, 0]]):
            rsa_mouse_dict['r'].extend(
                [stats.pearsonr(modelmat[i_code].flatten(), datamat.flatten())[0] for i_code in range(n_codes)])
    rsa_mouse_df = pd.DataFrame(rsa_mouse_dict)

    for index, row in ret_df.iterrows():
        rsa_dict['name'].extend([row['name']] * n_codes * 3)
        rsa_dict['figure_path'].extend([row['figure_path']] * n_codes * 3)
        rsa_dict['code'].extend(code_order * 3)
        rsa_dict['i_code'].extend(list(range(n_codes)) * 3)
        rsa_dict['metric'].extend(np.repeat(metrics, n_codes))
        for metric, modelmat, datamat in zip(metrics,
                                             [all_dists[0, :, 0], all_dists[0, :, 1], all_rda[0, :]],
                                             [pairwise_dists[index, 0], pairwise_dists[index, 1], rda[index]]):
            rsa_dict['r'].extend(
                [stats.pearsonr(modelmat[i_code].flatten(), datamat.flatten())[0] for i_code in range(n_codes)])
    rsa_df = pd.DataFrame(rsa_dict)

    stars = {k: {} for k in metrics}
    for metric in metrics[:2]:
        print(metric)
        metric_df = rsa_df[rsa_df['metric'] == metric]
        model = mixedlm('r ~ C(i_code)', metric_df, groups='name')
        mfit = model.fit(method=['powell', 'lbfgs'], maxiter=2000)
        print(mfit.summary())
        stars[metric][0] = np.arange(1, n_codes)
        pvals = mfit.summary().tables[1][-n_codes:-1]['P>|z|'].values
        stars[metric][1] = [np.float64(x) if x != '' else 1. for x in pvals]

        metric_mouse_df = rsa_mouse_df[rsa_mouse_df['metric'] == metric]
        anova_res = AnovaRM(data=metric_mouse_df, depvar='r', subject='name', within=['code']).fit()
        # print(class_label)
        print(anova_res)
        stars[metric][2] = [n_codes / 2]
        stars[metric][3] = [anova_res.anova_table["Pr > F"][0]]

    plt.figure(figsize=(4, 3))
    rsa_df_mouseavg = rsa_df.groupby(['name', 'code', 'i_code', 'metric'], as_index=False).mean()

    grid_kwargs = dict(col='metric', col_order=metrics[:2], aspect=(n_codes + 8) / 10, height=2, sharex=False,
                       gridspec_kws={'wspace': 0.1, 'hspace': 0.5})
    hue_kwargs = dict(x='i_code', y='r', hue='name', palette=mouse_colors, zorder=1, legend=False)
    mean_kwargs = dict(x='i_code', y='r', hue='i_code', palette=model_palette, errwidth=4, errorbar=('ci', 95))

    for i, df in enumerate([rsa_df_mouseavg, rsa_mouse_df]):
        g = sns.FacetGrid(data=df, **grid_kwargs)  # [df['code'] != 'Partial Distributed AU']
        g.map_dataframe(sns.lineplot, estimator=None, **hue_kwargs)  # style='dec_key', style_order=list(style_order),
        g.map_dataframe(sns.pointplot, **mean_kwargs).set_titles("")  # "{col_name}")
        g.set(ylim=(0, 1))

        for i_col, metric in enumerate(metrics[:2]):
            ax = g.axes[0, i_col]
            ax.set_xticklabels(code_order, rotation=45, rotation_mode='anchor', ha='right')
            ax.set_xlabel('')
            ax.set_title(metric)
            # print(len(stars[metric][i*2]), stars[metric][i*2])
            # print(len(stars[metric][i*2 + 1]), stars[metric][i*2 + 1])
            plot_stars(ax, stars[metric][i*2], stars[metric][i*2 + 1])

        g.axes[0, 0].set_ylabel(f'{protocol}:\nPearson correlation')
        hide_spines()
        plt.savefig(f'figs/{protocol}_pearson_corr_pc_dists_{i}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'figs/{protocol}_pearson_corr_pc_dists_{i}.pdf')


