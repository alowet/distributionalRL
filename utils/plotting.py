import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
import numpy as np
from scipy.stats import sem, friedmanchisquare, wilcoxon
import cmocean
import warnings
from protocols import get_cs_info
from paths import raise_print
from db import get_db_info, select_db

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def hide_spines():
    """
    Hides the top and rightmost axis spines from view for all active
    figures and their respective axes.
    Retrieved on 3/12/20 from https://stackoverflow.com/questions/3439344/setting-spines-in-matplotlibrc
    """
    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # for axis in ['top','bottom','left','right']:
            # ax.spines['left'].set_linewidth(0.5)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_stars(ax, centers, ps, star_color=None, font='DejaVu Sans', s=10, bonferroni=False, ytop_scale=0.99, show_ns=False, show_cross=False):

    assert(len(centers) == len(ps))
    n_grps = len(centers)
    
    if bonferroni:
        n_corr = n_grps
    else:
        n_corr = 1
        
    ytop = ax.get_ylim()[1]*ytop_scale
    
    if star_color is None:
        star_color = ['k'] * n_grps
    elif type(star_color) != list:
        star_color = [star_color] * n_grps 
    else:
        assert(len(star_color) == n_grps)

    for i, (cent, p) in enumerate(zip(centers, ps)):
        ax.text(cent, ytop, get_stars(p, n_corr, show_ns, show_cross), color=star_color[i], ha='center', fontdict={'family': font, 'size': s})


def get_stars(p, n_corr=1, show_ns=False, show_cross=False):
    if p < .001 / n_corr:
        stars = '∗∗∗'
    elif p < .01 / n_corr:
        stars = '∗∗'
    elif p < .05 / n_corr:
        stars = '∗'
    elif p < .10 / n_corr and show_cross:
        stars = '+'
    elif show_ns:
        stars = 'n.s.'
    else:
        stars = ''
    return stars


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    """
    https://stackoverflow.com/questions/23528477/share-axes-in-matplotlib-for-only-part-of-the-subplots
    """
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        try:
            if sharex:
                target._shared_axes['x'].join(target, ax)
            if sharey:
                target._shared_axes['y'].join(target, ax)
        except AttributeError:  # Matplotlib 3.3 has different API
            if sharex:
                target._shared_x_axes.join(target, ax)
            if sharey:
                target._shared_y_axes.join(target, ax)
                
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex:
        if axs.ndim > 1:
            for ax in axs[:-1,:].flat:
                ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
                ax.xaxis.offsetText.set_visible(False)
        else:
            for ax in axs[:-1].flat:
                ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
                ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey:
        if axs.ndim > 1:
            for ax in axs[:,1:].flat:
                ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
                ax.yaxis.offsetText.set_visible(False)
        else:
            for ax in axs[1:].flat:
                ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
                ax.yaxis.offsetText.set_visible(False)


def del_unused_axes(fig, axs, active_cols):
    for i_col in range(len(axs[0, :])):
        for i_row in range(len(axs[:, 0])):
            if i_col not in active_cols:
                fig.delaxes(axs[i_row, i_col])

def add_cbar(fig, last_pcolor, cbar_label, pad=15, d3=False, width=.01, left=.83):
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.2, hspace=0.2)
    if d3:
        cbar_ax = fig.add_axes([.97, 0.38, 0.03, 0.4])
    else:
        cbar_ax = fig.add_axes([left, 0.38, width, 0.4])
    cbar = fig.colorbar(last_pcolor, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=pad)
    return cbar_ax

def add_vlines(fig, x2, exclude_axs=[], cbar_ax=None, lw=2, vline_color=[0.25, 0.25, 0.25]):
    for ax in fig.axes:
        if ax is not cbar_ax:
            ax.axvline(x=0, c=vline_color, lw=lw)
            if ax not in exclude_axs:
                ax.axvline(x=x2, c=vline_color, lw=lw)

def add_cbar_and_vlines(fig, last_pcolor, cbar_label, x2, exclude_axs=[], lw=2, vline_color=[0.25, 0.25, 0.25]):
    cbar_ax = add_cbar(fig, last_pcolor, cbar_label)
    add_vlines(fig, x2, exclude_axs, cbar_ax, lw, vline_color)
    hide_spines()

def plot_ctd(psth_bins, delay_to_plot, mat, label, kwargs={}, pad=15):
    fig, axs = plt.subplots(figsize=(5, 4))
    im = plt.pcolormesh(psth_bins[delay_to_plot], psth_bins[delay_to_plot], mat, **kwargs)        
    plt.vlines([0, 1, 3], -1, 5, 'w')
    plt.hlines([0, 1, 3], -1, 5, 'w')
    plt.xlim(psth_bins[delay_to_plot][0], psth_bins[delay_to_plot][-1])
    plt.ylim(psth_bins[delay_to_plot][-1], psth_bins[delay_to_plot][0])
    plt.yticks([0, 1, 2])
    plt.ylabel('Training time (s)')
    plt.xlabel('Testing time (s)')
    add_cbar(fig, im, label, pad)


def plot_avgs(n_trial_types, time_toplot, timecourse, quant1, quant2, colors, trace_type_names, trace_start=1., trace_end=3., plot_indiv=True, show_stats=True):
    """
    Plot averages across levels, where levels is mice, sessions, or neurons
    :param n_trial_types: Usually 6
    :param time_toplot: 1D vector, timebase
    :param timecourse: 3D array, n_trial_types x n_levels x len(time_toplot)
    :param quant1: 2D array, n_trial_types x n_levels, first item to quantify, e.g. mean licking rate during last 0.5 s trace
    :param quant2: 2D array, n_trial_types x n_levels, second item to quantify, e.g. cumulative # licks during trace
    :param colors: list of colors, len = n_trial_types
    :param trace_start: generally 1s, for drawing vertical line
    :param trace_end: generally 3s, for drawing vertical line
    :param plot_indiv: plot individual connected lines for each level (session or animals)
    :return:
    """
    stderr_timecourse = sem(timecourse, axis=1)
    mean_timecourse = np.mean(timecourse, axis=1)

    stderr1 = sem(quant1, axis=1)
    mean1 = np.mean(quant1, axis=1)

    stderr2 = sem(quant2, axis=1)
    mean2 = np.mean(quant2, axis=1)

    # set up figure
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.5), gridspec_kw={'wspace': 0.3})
    line_kwargs = {'color': [.7, .7, .7], 'alpha': .2}
    error_kwargs = {'fmt': 'none', 'color': colors, 'lw': 4, 'zorder': 5}
    scatter_kwargs = {'c': colors, 's': 50, 'alpha': 1, 'zorder': 5}
    type_range = range(n_trial_types)

    # plot timecourse
    ax = axs[0]
    for i_type in type_range:
        ax.plot(time_toplot, mean_timecourse[i_type, :], c=colors[i_type], lw=4)
        ax.fill_between(time_toplot, mean_timecourse[i_type, :] - stderr_timecourse[i_type, :],
                        mean_timecourse[i_type, :] + stderr_timecourse[i_type, :],
                        color=colors[i_type], ec=colors[i_type], alpha=.2)
    ax.axvspan(0, np.mean(trace_start), alpha=.5, color=(.8, .8, .8))
    ax.axvline(np.mean(trace_end), ls='--', alpha=.5, color=(.5, .5, .5))
    #     ax.set_ylabel('Lick rate (Hz)')
    ax.set_xlabel('Time from CS (s)')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))

    # plot mean lick rate in 0.5 s before reward
    ax = axs[1]
    if plot_indiv: ax.plot(type_range, quant1, '-o', **line_kwargs)
    ax.errorbar(type_range, mean1, stderr1, **error_kwargs)
    ax.scatter(type_range, mean1, **scatter_kwargs)
    #     ax.set_ylabel('Avg. lick rate during last 0.5s (Hz)')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(type_range)
    # ax.set_xticklabels(['CS{}'.format(i_type) for i_type in type_range])
    ax.set_xticklabels(trace_type_names)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))

    # plot cumulative number of licks in entire 2.0 s trace period
    ax = axs[2]
    if plot_indiv: ax.plot(type_range, quant2, '-o', **line_kwargs)
    ax.errorbar(type_range, mean2, stderr2, **error_kwargs)
    ax.scatter(type_range, mean2, **scatter_kwargs)
    #     ax.set_ylabel('Cumulative licks during trace')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(type_range)
    # ax.set_xticklabels(['CS{}'.format(i_type) for i_type in type_range])
    ax.set_xticklabels(trace_type_names)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))

    if show_stats:
        # perform statistics on these session means
        n_comps = (n_trial_types ** 2 - n_trial_types) / 2
        for i, arr in enumerate([quant1, quant2]):
            print(axs[i + 1].get_ylabel())
            try:
                print(friedmanchisquare(*arr))
                for j in range(n_trial_types):
                    for k in range(j):
                        stat, p = wilcoxon(arr[j, :], arr[k, :])
                        # print('{} vs {} w/ Bonferroni: p = {:.4f}'.format(k, j, min(p * n_comps, 1)))
                        print('{} vs {}: p = {:.4f}'.format(k, j, p))
            except ZeroDivisionError:
                print(arr) 
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    hide_spines()
    return axs

def set_font_size(params, lg_font_size, sm_font_size):
    params['axes.titlesize'] = lg_font_size
    params['axes.labelsize'] = lg_font_size
    params['legend.fontsize'] = sm_font_size
    params['xtick.labelsize'] = sm_font_size
    params['ytick.labelsize'] = sm_font_size
    matplotlib.rcParams.update(params)
    return params

def get_timebase(std_time, native_time, frame_period, decround):
    time_toplot = np.arange(std_time[0], std_time[-1], frame_period)

    # index of first and last licking samples to plot
    start_align = int(np.flatnonzero(np.equal(np.round(native_time, decround), np.round(time_toplot[0], decround))))
    end_align = int(np.flatnonzero(np.equal(np.round(native_time, decround), np.round(time_toplot[-1], decround))))

    return time_toplot, start_align, end_align

def validate_timestamps(timestamps):
    for dur in ['trace', 'stim']:
        tmp = timestamps[dur][~np.isnan(timestamps[dur])]
        timestamps[dur + '_trial'] = timestamps[dur].copy()
        if not np.all(np.around(tmp, 2) == np.around(tmp[0], 2)):
            raise_print("Not all " + dur + " are equal. Figure out how you'd like to handle this (e.g. time warping?)")
        timestamps[dur] = np.around(tmp[0], 2)
        timestamps[dur + '_trial'][np.isnan(timestamps[dur + '_trial'])] = 0
    return timestamps

def plot_all_neurons(activity, behavior, pupil, trial_types, protocol, timestamps, pcolor_time, db_dict, names_tosave,
                     label, chan_name, cmap=None, first_trial=0, lw=2):

    colors, _, trial_type_names, _, n_trial_types, n_trace_types, _ = get_cs_info(protocol)
    n_cells = activity.shape[0]

    prc_range = np.nanpercentile(activity, [2.5, 97.5], axis=None)
    if not cmap:
        try:
            cmap = cmocean.tools.crop(cmocean.cm.balance, prc_range[0], prc_range[1], pivot=0)
        except AssertionError:
            cmap = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grand_mean = np.nanmean(activity, axis=0)
        grand_sem = sem(activity, axis=0)

    # set up all neuron figure
    if pupil:
        fig, axs = plt.subplots(4, n_trial_types, figsize=(15, 12.5),
                                           gridspec_kw={'height_ratios': [1, 4, 1, 1]}, sharex=True)
    else:
        fig, axs = plt.subplots(3, n_trial_types, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 4, 1]},
                                           sharex=True)
    [set_share_axes(axs[i, :], sharey=True) for i in range(axs.shape[0])]
    del_unused_axes(fig, axs, np.unique(trial_types))

    trial_inds_all_types = [np.flatnonzero(trial_types == i) for i in range(n_trial_types)]
    for i_type in range(n_trial_types):

        # first row: average licking trace
        ax = axs[0, i_type]
        trial_type_inds_beh = trial_inds_all_types[i_type] + first_trial
        # licks_this_type = behavior['dat']['licks_smoothed'][trial_types == i_type,
        #                   behavior['start']:behavior['end'] + 1]
        licks_this_type = behavior['dat']['licks_smoothed'][trial_type_inds_beh, behavior['start']:behavior['end'] + 1]
        mean_licking_pattern = np.mean(licks_this_type, axis=0)
        sem_licking_pattern = sem(licks_this_type, axis=0)

        ax.plot(behavior['time'], mean_licking_pattern, color=colors[i_type])
        ax.fill_between(behavior['time'], mean_licking_pattern + sem_licking_pattern, mean_licking_pattern -
                        sem_licking_pattern, color=colors[i_type], edgecolor=None, alpha=0.2)
        ax.set_title(trial_type_names[i_type])
        ax.set_ylim(0, 15)
        if i_type == 0:
            ax.set_ylabel('Lick rate (Hz)')

        # second row: heatmap, one line for each neuron
        ax = axs[1, i_type]
        im = ax.pcolormesh(pcolor_time, np.arange(n_cells + 1), activity[:, i_type, :], vmin=prc_range[0],
                           vmax=prc_range[1], cmap=cmap)
        if i_type == 0:
            ax.set_ylabel('Neuron #')
        ax.set_ylim(n_cells, 0)

        # third row: average response, averaged across neurons
        ax = axs[2, i_type]
        ax.plot(timestamps['time'], grand_mean[i_type], color=colors[i_type], lw=lw)
        ax.fill_between(timestamps['time'], grand_mean[i_type, :] + grand_sem[i_type, :],
                        grand_mean[i_type, :] - grand_sem[i_type, :], color=colors[i_type], alpha=0.2)
        if i_type == 0:
            ax.set_ylabel('Grand Mean ' + label)
        ax.set_xticks((0, timestamps['stim'] + timestamps['trace'] + timestamps['iti']))
        ax.set_xlabel('Time from CS (s)')

        # fourth row (if pupil_data exists): delta pupil size, averaged across trials
        if pupil:
            pupil_radius = pupil['dat']['pupil_mat'][trial_type_inds_beh, pupil['start']:pupil['end'] + 1]
            mean_radius = np.mean(pupil_radius, axis=0)
            sem_radius = sem(pupil_radius, axis=0)

            ax = axs[3, i_type]
            ax.plot(pupil['time'], mean_radius, color=colors[i_type], lw=lw)
            ax.fill_between(pupil['time'], mean_radius + sem_radius, mean_radius -
                            sem_radius, color=colors[i_type], edgecolor=None, alpha=0.2)
            if i_type == 0:
                ax.set_ylabel(r'Mean ($\Delta$ Radius)/Radius')

    fig.suptitle(db_dict['name'] + ' ' + db_dict['file_date_id'])
    add_cbar_and_vlines(fig, im, 'Mean FR (std)', timestamps['stim'] + timestamps['trace'])
    [fig.savefig('_'.join([froot, chan_name, label]) + '.png', bbox_inches='tight', dpi=300) for froot in names_tosave['filenames']]

def plot_confusion(conf, ax, name, normalize=False):
    if normalize:
        conf = conf / np.sum(conf, axis=-1, keepdims=True)
    im = ax.imshow(conf, vmin=0, vmax=1, cmap='magma')  # plot normalized confusion matrix
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Predicted Label')

    return im

def plot_box(means, shuff, names, ps=[], vline_color=[0.25, 0.25, 0.25]):
    """
    means: n x n_clf array of scores, where n = n_rets or n_folds
    shuff: n x n_clf array of scores of trial shuffle, where n = n_rets or n_folds, OR integer, = n_active_types
    """
    # boxplot algorithm comparison
    plt.figure(figsize=(10, 8))
    plt.boxplot(means, notch=True)
    if isinstance(shuff, int):
        plt.axhline(1./shuff, ls='--', c=[.4, .4, .4])
    else:
        n_clf = means.shape[1]
        plt.errorbar(np.arange(1, n_clf + 1), np.mean(shuff, axis=0), sem(shuff, axis=0) * 1.96, c=vline_color,
                     fmt='none', lw=4)
    plt.xticks(ticks=range(len(names)+1), labels=[''] + names, rotation=90)
    plt.ylabel('Balanced accuracy')

    for i, p in enumerate(ps):
        if p < .001:  # / n_clf:  # Bonferroni corrected
            plt.text(i+1, plt.ylim()[1], '***', fontsize='large', ha='center')
        elif p < .01:  # / n_clf:  # Bonferroni corrected
            plt.text(i+1, plt.ylim()[1], '**', fontsize='large', ha='center')
        elif p < .05:  # / n_clf:  # Bonferroni corrected
            plt.text(i+1, plt.ylim()[1], '*', fontsize='large', ha='center')

    hide_spines()


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    """
    https://matplotlib.org/devdocs/gallery/lines_bars_and_markers/scatter_hist.html
    """
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    bins = np.arange(0, lim, lim / 20)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
    return lim


def label_not_nan(arr, label):
    if np.isnan(arr).any():
        return f'_{label}'
    return label


def summarize_behavior(last_sec_means, n_trace_types, protocol_info, colors, po, savename, unit,
    tnames=None, tpos=None, mouse_colors=None, hline=None):

    # set up figure
    fig, ax = plt.subplots(figsize=(2.3, 1.7))
    if mouse_colors is None:
        line_kwargs = {'color': [.7, .7, .7], 'alpha': .2}
    else:
        line_kwargs = {}
        ax.set_prop_cycle('color', mouse_colors.values())

    # error_kwargs = {'fmt': 'none', 'color': [colors['prot_color']]*n_trace_types, 'lw': 4, 'zorder': 5}
    # scatter_kwargs = {'c': [colors['prot_color']]*n_trace_types, 's': 50, 'alpha': 1, 'zorder': 5}
    error_kwargs = {'fmt': 'none', 'lw': 3, 'zorder': 5, 'ecolor': colors['colors'][:n_trace_types][po]}
    scatter_kwargs = {'c': colors['colors'][:n_trace_types][po], 's': 50, 'alpha': 1, 'zorder': 5}

    gr_se = sem(last_sec_means, axis=1)
#     gr_sd = np.std(last_sec_means, axis=1)
    gr_mean = np.mean(last_sec_means, axis=1)

    # ax.plot(np.arange(n_trace_types), last_sec_means, '-o', **line_kwargs)
    ax.plot(np.arange(n_trace_types), last_sec_means, **line_kwargs, zorder=1)
#     ax.set_prop_cycle(cycler('color', colors['colors'][:n_trace_types]))
    ax.errorbar(np.arange(n_trace_types), gr_mean, gr_se * 1.96, **error_kwargs)
    ax.scatter(np.arange(n_trace_types), gr_mean, **scatter_kwargs)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    
    # ax.set_xticklabels(trace_type_names, fontsize=16)
    if tnames is not None:
        ax.set_xticks(tpos)
        ax.set_xticklabels(tnames)
    else:
        ax.set_xticks(np.arange(n_trace_types))
        ax.set_xticklabels(protocol_info['trace_type_names'])
    # ax.axis["bottom"].major_ticklabels.set_ha("right")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))
    # ax.set_ylabel('Avg. anticipatory\nlick rate (Hz)')
    ax.set_ylabel('Anticipatory\nlick rate (Hz)')
    hide_spines()
    ax.spines['left'].set_position(("axes", -0.1))

    if hline is not None:
        ax.axhline(hline, ls='--', color=[.5] * 3, lw=1, zorder=0)
    else:
        value_ax = ax.twinx()
        value_color = 'k'
        # value_ax.set_ylabel('Expected\nreward ($\mu$L)', color=value_color, rotation=270,  labelpad=60)
        value_ax.plot(np.arange(n_trace_types), np.array(protocol_info['mean'])[po], ls='--', color=value_color, zorder=1)
        value_ax.tick_params(axis='y', labelcolor=value_color)
        value_ax.set_yticks([0, 4])
        value_ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))
        # value_ax.set_ylim([-2, 16])

        ymax = (protocol_info['mean'][po[-1]] * last_sec_means.max() * 1.1) / last_sec_means[-1].mean()
        value_ax.set_ylim([-1, ymax])
        value_ax.spines['top'].set_color('none')
        value_ax.spines['left'].set_color('none')
        value_ax.spines['right'].set_position(("axes", 1.1))
        value_ax.set_ylabel('Expected\nreward ($\mu$L)')

    print(savename)

    plt.savefig(savename + '_' + unit + '.pdf', bbox_inches='tight')
    plt.savefig(savename + '_' + unit + '.svg', bbox_inches='tight')
    plt.savefig(savename + '_' + unit + '.png', bbox_inches='tight', dpi=500)

def get_mouse_colors(protocol, imaging=False):

    # this method ensures the same color mapping across lesion and helper 
    paths = get_db_info()
    if imaging:
        mice_rets = select_db(paths['db'], 'session', 'name', 'protocol=? AND has_imaging=1 AND significance=1 AND has_facemap=1', (protocol.replace('DiverseDists', 'DistributionalRL_6Odours'),), unique=False)
    else:
        mice_rets = select_db(paths['db'], 'session', 'name', 'protocol=? AND has_ephys=1 AND significance=1 AND has_facemap=1', (protocol.replace('DiverseDists', 'DistributionalRL_6Odours'),), unique=False)
    all_mice = sorted(np.unique([ret['name'] for ret in mice_rets]))

    try:
        if imaging:
            color_dict = {'SameRewDist': 'Dark2',  # 8 mice
                  'DistributionalRL_6Odours': 'Set2',  # for backward compatibility
                  'DiverseDists': 'Set2',
                  'Bernoulli': 'Set1',
                  }
        else:
            color_dict = {'SameRewDist': 'Set3',  # 12 mice
                  'SameRewVar': 'Pastel1', # 5-6 mice
                  'DistributionalRL_6Odours': 'Pastel2',  # for backward compatibility
                  'DiverseDists': 'Pastel2',
                  'Bernoulli': 'Accent',
                  'SameRewSize': 'tab10'
                  }
        color_set = mpl.cm.get_cmap(color_dict[protocol]).colors
    except KeyError:
        raise Exception('Protocol not found')
    mouse_colors = {k: v for k, v in zip(all_mice, color_set)}

    return mouse_colors