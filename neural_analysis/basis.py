import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../utils')
from plotting import hide_spines


# Translated to Python from https://github.com/pillowlab/GLMspiketools/blob/master/glmtools_misc/makeBasis_PostSpike.m
# nonlinearity for stretching x axis (and its inverse)
nlin = lambda x: np.log(x + 1e-20)
invnl = lambda x: np.exp(x) - 1e-20  # inverse nonlinearity
ff = lambda x, c, dc: (np.cos(
    np.maximum(-np.pi, np.minimum(np.pi, (x - c) * np.pi / dc / 2))) + 1) / 2  # raised cosine basis vector


# evaluate the predictors at the basis functions
def eval_cosine_bump(x, center, width):
    x_reshape = x.reshape(-1, )
    y = (np.cos(2 * np.pi * (x_reshape - center) / width) * 0.5 + 0.5) * (np.absolute(x_reshape - center) < width / 2)
    return y


def create_basis(n_samps, n_bases):
    # linearly space the centers of the position bases
    centers = np.linspace(0, n_samps, n_bases)
    # find the spacing between two adjacent centers
    spacing = np.mean(np.diff(centers))
    # set width of the position bases
    widths = 4 * spacing
    # create a linear spaced position series
    times = np.linspace(0, n_samps, n_samps)
    # evaluate the values of the position series on each base
    bases = np.full((n_samps, n_bases), np.nan)
    for idx, cent in enumerate(centers):
        bases[:, idx] = eval_cosine_bump(times, cent, widths)

    # visualize
    plt.plot(times, bases)
    plt.xlabel('Time')
    hide_spines()

    return bases, widths, centers


def plot_tuning_funcs(bases, regressor_labels, coefs, n_back, n_bases, n_types, nsamps_per_trial, save_name):
    plt.figure()
    labels = ['{}-back reward magnitudex '.format(i) for i in range(1, n_back + 1)] + ['']

    for lab in labels:
        # identify indices for position bases
        start_idx = regressor_labels.index('{}tm_bump0'.format(lab)) + 1  # plus 1 for intercept term
        end_idx = regressor_labels.index('{}tm_bump{}'.format(lab, n_bases - 1)) + 1

        # grab corresponding coefficient
        coef = coefs[start_idx:end_idx + 1]
        # reconstruct tuning function
        tuning = np.sum((bases * coef), axis=1)

        # create a linear spaced position series
        times = np.linspace(-1, 5, nsamps_per_trial)
        plt.plot(times, tuning, label=lab + '.')
        plt.xlabel('Time')
        plt.legend(bbox_to_anchor=(1.02, 1))
        plt.title('Kernels for all expanded variables')
    hide_spines()
    chkdir(save_name)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')


def plot_kernels(filts, conv_base_labels, regressor_labels, coefs, n_filts, filt_time, save_name):
    ncols = 3
    nrows = int(np.ceil(len(conv_base_labels) / 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    # plot temporal kernels
    for i_lab, lab in enumerate(conv_base_labels):
        ax = axs.flat[i_lab]
        start_idx = regressor_labels.index('{}*tm_filt0'.format(lab)) + 1
        end_idx = regressor_labels.index('{}*tm_filt{}'.format(lab, n_filts - 1)) + 1
        coef = coefs[start_idx:end_idx + 1]
        ax.plot(filt_time, np.sum((filts * coef), axis=1), 'k')
        ax.set_xlabel('Time (sec)')
        ax.set_title(lab)
    hide_spines()
    chkdir(save_name)
    fig.savefig(save_name, dpi=300, bbox_inches='tight')


def plot_coefs(coefs, regressor_labels, save_name):
    plt.figure(figsize=(20, 10))
    coefs = np.array(coefs)
    rl = np.array(regressor_labels, dtype=object)
    coef_bool = np.abs(coefs[1:]) > 0
    rl_use = rl[coef_bool]
    ticks = np.flatnonzero(coef_bool)
    n_regressors = len(regressor_labels)
    plt.scatter(np.arange(n_regressors), coefs[1:])
    plt.xticks(ticks, rl_use, rotation=90)
    plt.ylabel('Coefficient')
    hide_spines()
    chkdir(save_name)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')


def plot_predictions(y, pred, save_name):
    # display the first 2000 timepoints
    plt.figure(figsize=(16, 2))
    plt.plot(y[:2000], 'k', linewidth=0.5, label='data')
    plt.plot(pred[:2000], 'r', label='prediction')
    plt.legend()
    hide_spines()
    chkdir(save_name)
    plt.savefig(save_name, bbox_inches='tight', dpi=300)


def chkdir(save_name):
    dir_name = os.path.dirname(save_name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

