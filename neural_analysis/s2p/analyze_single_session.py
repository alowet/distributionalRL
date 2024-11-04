import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import Isomap
from sklearn.cluster import SpectralClustering
from scipy.stats import zscore, wilcoxon, ttest_rel, f_oneway, kruskal, linregress, spearmanr, friedmanchisquare
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import minimize
from scipy import odr
import tensortools as tt
import cmocean
import pickle
import dill
import os
from analysisUtils import *
from neuralUtils import *
from pop_code import Bernoulli, PopCode

sys.path.append('../utils')
from db import get_db_info, create_connection, select_db
from plotting import plot_avgs, hide_spines
from paths import parse_data_path, check_dir
from matio import loadmat
from protocols import load_params

sys.path.append('../behavior_analysis')
from traceUtils import check_stats

def analyze_single_session(mouse_name, file_date_id, protocol, table):

	file_date = file_date_id[:8]
	if table == 'ephys':
		activity_type = 'firing'  # use smoothed activity
		suffix = 'spikes'
	else:
		activity_type = 'spks'
		suffix = 'Ca'

	paths = get_db_info()
	fig_dir = os.path.join(paths['neural_fig_roots'][0], mouse_name, file_date_id, 'session')
	check_dir(fig_dir)
	data_path = os.path.join(paths['neural_fig_roots'][0], mouse_name, file_date_id, '_'.join([mouse_name, file_date, suffix+'.p']))

	# general parameters
	colors, protocol_info, periods, _ = load_params(protocol)
	exclude_tt = protocol_info['exclude_tt']
	activity_label = get_activity_label(activity_type)

	# extract X and y from file
	cached_data = pickle.load(open(data_path, 'rb'))
	activity = cached_data[activity_type]
	n_cells = np.shape(activity)[0]
	all_trials = np.shape(activity)[1]

	timestamps = cached_data['timestamps']  # for compatibility with imaging sessions
	if 'first_trial' not in timestamps:
		timestamps['first_trial'] = 0
	if 'last_trial' not in timestamps:
		timestamps['last_trial'] = all_trials

	odor_on_idx = np.argmin(np.abs(timestamps['time']))
	trace_start_idx = np.argmin(np.abs(timestamps['time'] - timestamps['stim']))
	trace_plus_1_idx = np.argmin(np.abs(timestamps['time'] - timestamps['stim'] - 1))
	trace_end_idx = np.argmin(np.abs(timestamps['time'] - timestamps['stim'] - timestamps['trace']))
	# go up to 1 second after reward delivery
	reward_end_idx = np.argmin(np.abs(timestamps['time'] - timestamps['stim'] - timestamps['trace'] - 1))

	# idx_pairs = [(0, odor_on_idx), (odor_on_idx, trace_start_idx), (trace_start_idx, trace_end_idx), (trace_end_idx, reward_end_idx)]
	idx_pairs = [(0, odor_on_idx), (odor_on_idx, trace_start_idx), (trace_start_idx, trace_plus_1_idx),
	             (trace_plus_1_idx, trace_end_idx), (trace_end_idx, reward_end_idx)]
	# IMPORTANT! Make sure to set indices appropriately based on idx_pairs, above
	# odor_ind = 1
	# early_trace_ind = 2
	late_trace_ind = 3
	main_comp_ind = late_trace_ind
	odor_comp_ind = 0
	trace_comp_ind = 2

	# get corresponding behavior data from behavior file
	db_entry = select_db(paths['db'], table, 'behavior_path', 'name=? AND file_date_id=?', (mouse_name, file_date_id))
	behavior_path = db_entry['behavior_path'].replace(paths['remote_behavior_root'], paths['behavior_root'])
	converted_data = loadmat(behavior_path)
	session_data = converted_data['SessionData']
	all_reward = np.array(session_data['RewardDelivered'][timestamps['first_trial']:timestamps['last_trial']],  dtype=np.int8)
	trial_types = session_data['TrialTypes'][timestamps['first_trial']:timestamps['last_trial']] - 1

	trial_inds_all_types = cached_data['trial_inds_all_types']
	all_active_types = np.unique(trial_types)
	active_types = [x for x in all_active_types if x not in exclude_tt]
	n_active_types = len(active_types)

	# # rearrange CS order if necessary
	# This is unfinished. In order for this to work, I would need to adjust many of the fields in protocol_info,
	# including mean_rews, norm_dists, var_types, low_tt. This should be done inside the function itself, but
	# it would be unsafe to do so now without making sure it doesn't mess up preliminary analysis
	# trial_types = np.array([cs_map[x-1] for x in trial_types])
	# rearranged_colors = [[]] * protocol_info['n_trial_types']
	# rearranged_palette = [[]] * protocol_info['n_trial_types']
	# trial_inds_all_types = [[]] * protocol_info['n_trial_types']
	# for i in range(protocol_info['n_trial_types']):
	#     rearranged_colors[cs_map[i]] = colors['colors'][i]
	#     rearranged_palette[cs_map[i]] = colors['palette'][i]
	#     trial_inds_all_types[cs_map[i]] = cached_data['trial_inds_all_types'][i]
	# colors['colors'] = rearranged_colors
	# colors['palette'] = rearranged_palette

	# remove unexpected reward trials, or else they will cause trouble
	activity = activity[:, [x not in exclude_tt + 1 for x in trial_types], :]
	n_trials = np.shape(activity)[1]

	n_samps_per_trial = reward_end_idx
	alpha = periods['alpha']

	### PERFORM TCA ON SINGLE SESSION ###
	# For TCA, assume N neurons at T time points in each trial, and that there are K total trials.
	# A natural way to represent this data is a three-dimensional data array with dimensions N x T x K.
	# This isn't strictly necessary; we just have to keep track of which dimension is which down the road when
	# analyzing our factors

	# downsample spike data so that it doesn't take forever to run TCA
	if table == 'ephys':
		ds_fun = interp1d(timestamps['time'], activity, axis=2)
		ds_time = np.arange(timestamps['time'][0], 5, 0.05)
		ds_activity = ds_fun(ds_time)
		ds_tca = np.swapaxes(ds_activity, 1, 2)
	else:
		ds_tca = (np.swapaxes(activity, 1, 2) - activity.min()) / (activity.max() - activity.min())
	# ds_tca is now N x T x K
	print(activity.shape, ds_tca.shape, ds_tca.min(), ds_tca.max())

	# TCA: https://github.com/ahwillia/tensortools
	# Fit ensembles of tensor decompositions.
	methods = (
		# 'cp_als',  # fits unconstrained tensor decomposition.
		'ncp_bcd',  # fits nonnegative tensor decomposition.
		'ncp_hals',  # fits nonnegative tensor decomposition.
	)

	ensembles = {}
	for m in methods:
		ensembles[m] = tt.Ensemble(fit_method=m, fit_options=dict(tol=1e-4))
		ensembles[m].fit(tcaF, ranks=range(1, min(5, n_cells + 1)), replicates=3)

	# Plotting options for the unconstrained and nonnegative models.
	plot_options = {
		'cp_als': {'line_kw': {'color': 'black', 'label': 'cp_als'}, 'scatter_kw': {'color': 'black'}},
		'ncp_hals': {'line_kw': {'color': 'blue', 'alpha': 0.5, 'label': 'ncp_hals'},
		             'scatter_kw': {'color': 'blue', 'alpha': 0.5}},
		'ncp_bcd': {'line_kw': {'color': 'red', 'alpha': 0.5, 'label': 'ncp_bcd'},
		            'scatter_kw': {'color': 'red', 'alpha': 0.5}}
	}

	# Plot similarity and error plots.
	plt.figure()
	for m in methods:
		tt.plot_objective(ensembles[m], **plot_options[m])
	plt.legend()
	plt.savefig(os.path.join(fig_dir, 'TCA_objective.png'), dpi=300, bbox_inches='tight')

	plt.figure()
	for m in methods:
		tt.plot_similarity(ensembles[m], **plot_options[m])
	plt.legend()
	plt.savefig(os.path.join(fig_dir, 'TCA_similarity.png'), dpi=300, bbox_inches='tight')

	# plot factors
	n_factors = 3
	trial_inds = np.full(all_trials, np.nan)
	for i_type in active_types:
		trial_inds[trial_inds_all_types[i_type]] = i_type
	# remove the NaNs, which are non-trace trials (unexpected reward)
	reward = all_reward[~np.isnan(trial_inds)]  # also remove from the reward vector
	trial_inds = trial_inds[~np.isnan(trial_inds)].astype(np.int8)
	trial_colors = [colors['colors'][i_trial] for i_trial in trial_inds]
	trial_inds_iso = np.repeat(trial_inds, n_samps_per_trial)

	markers = np.array(['o', 'v', '^', 's', 'p', '+', 'x', '*'])
	trial_markers = markers[reward.astype(np.int8)]

	for m in methods:
		fig, axs = plt.subplots(n_factors, 3, figsize=(14, 3 * n_factors), squeeze=False)
		for i_fact in range(n_factors):
			neuron_factors = ensembles[m].results[n_factors][0].factors[0][:, i_fact]
			time_factors = ensembles[m].results[n_factors][0].factors[1][:, i_fact]
			trial_factors = ensembles[m].results[n_factors][0].factors[2][:, i_fact]
			axs[i_fact, 0].bar(range(n_cells), neuron_factors)
			axs[i_fact, 0].set_ylabel('Factor {}'.format(i_fact + 1))
			axs[i_fact, 1].plot(timestamps['time'], time_factors)
			axs[i_fact, 1].axvline(0)
			axs[i_fact, 1].axvline(timestamps['stimulus_dur'] + timestamps['trace_dur'])
			mscatter(x=range(n_trials), y=trial_factors, ax=axs[i_fact, 2], m=trial_markers, color=trial_colors)

		axs[0, 0].set_title('Neuron Factors')
		axs[0, 1].set_title('Temporal Factors')
		axs[0, 2].set_title('Across-Trial Factors')

		fig.tight_layout(rect=[0, 0.03, 1, 0.95])
		fig.suptitle(m)
		hide_spines()
		fig.savefig(os.path.join(fig_dir, 'factors_' + m + '.png'), dpi=300, bbox_inches='tight')

	# get new trial_inds_all_types, but with the unexpected reward trial numbers subtracted
	trial_inds_inc_types = [np.flatnonzero(trial_inds == x) for x in active_types]

	# X must be (nsamples, nfeatures) for sklearn
	# average across trial timecourses (within trial types)
	cell_means_tt = np.stack([np.mean(activity[:, tt_inds, :], axis=1) for tt_inds in trial_inds_inc_types], axis=2)
	print(cell_means_tt.shape)

	n_periods = periods['n_periods']
	n_comp_periods = periods['n_comp_periods']

	X_means = np.zeros((n_cells, n_trials, n_periods))
	X_tt_means = np.zeros((n_cells, n_periods, n_active_types))
	for i_pair, pair in enumerate(idx_pairs):
		X_means[:, :, i_pair] = np.mean(activity[:, :, pair[0]:pair[1]], axis=2)
		X_tt_means[:, i_pair, :] = np.mean(cell_means_tt[:, pair[0]:pair[1], :], axis=1)

	X_concat = np.reshape(activity[:, :, 0:reward_end_idx], (n_cells, n_trials * n_samps_per_trial))
	time_concat = np.tile(timestamps['time'][0:reward_end_idx], n_trials)
	epoch = np.repeat(a=[0, 1, 2, 3, 4],
	                  repeats=[odor_on_idx, trace_start_idx - odor_on_idx,
	                           trace_plus_1_idx - trace_start_idx, trace_end_idx - trace_plus_1_idx,
	                           reward_end_idx - trace_end_idx])
	epoch_concat = np.tile(epoch, n_trials)

	# deconstruct X_means into n_dists x n_cells x max_n_trials
	n_max_trials_per_type = np.amax([len(x) for x in trial_inds_inc_types])
	cue_resps = np.full((n_active_types, n_cells, n_max_trials_per_type), np.nan)
	for i_dist in range(n_active_types):
		cue_resps[i_dist, :, :len(trial_inds_inc_types[i_dist])] = X_means[:, trial_inds_inc_types[i_dist], main_comp_ind]

	fperiod = np.diff(timestamps['time'])[0]
	X_concat_pcolor_time = np.arange(0, (n_trials * n_samps_per_trial + 1) * fperiod, fperiod)
	X_concat_time = np.arange(0, (n_trials * n_samps_per_trial) * fperiod, fperiod)

	# check for reliable mean neural activity differences vs. baseline, irrespective of trial type
	modulation = np.zeros((n_cells, n_comp_periods))
	for i_period in range(1, n_periods):
		_, modulation[:, i_period - 1] = stats.ttest_rel(X_means[..., 0], X_means[..., i_period], axis=1)

	# check for reliable mean neural activity difference(s) across trial types for each cell
	tt_period_mean = [X_means[:, tt_inds, 1:] for tt_inds in trial_inds_inc_types]
	_, discrim = stats.f_oneway(*tt_period_mean, axis=1)  # different number of trials for each trial type

	modlu_inds = np.flatnonzero(modulation[:, trace_comp_ind] < alpha)
	discrim_inds = np.flatnonzero(discrim[:, trace_comp_ind] < alpha)
	print(modlu_inds, discrim_inds)

	# regress mean reward size on neural activity
	# grand_mean = np.stack([np.mean(x, axis=1) for x in tt_period_mean], axis=2)  # n_cells x n_comp_periods x n_tt
	rew_correls = find_rew_correls(X_tt_means[:, 1:, :], protocol_info['mean_rews'])
	rew_correl_inds = np.flatnonzero(
		rew_correls[:, odor_comp_ind, 3] < alpha)  # 3 is index where p value is returned by linregress
	print(rew_correl_inds)
	print(rew_correls[rew_correl_inds, odor_comp_ind, :])

	# Plot the mean response of each cell for each trial type. We expect there to be an increase for CS5 and decrease
	# for CS4 in the *population* of all cells, plotted together
	fig, axs = plt.subplots(1, n_periods, figsize=(12, 3))
	for i in range(n_periods):
		ax = axs[i]
		sns.swarmplot(data=X_tt_means[:, i, :], ax=ax, palette=colors['palette'])
		ax.set_title(periods['period_names'][i])
		print(friedmanchisquare(*X_tt_means[:, i, :].T))
		n_comps = .5 * n_active_types ** 2 - n_active_types
		for j in range(n_active_types):
			for k in range(j):
				stat, p = wilcoxon(X_tt_means[:, i, j], X_tt_means[:, i, k])
				print('{} vs {}: p = {:.4f}'.format(j, k, p))
	#             print('{} vs {} w/ Bonferroni: p = {:.4f}'.format(j, k, min(p*n_comps, 1)))
	#     print(pairwise_tukeyhsd(df_data['cell_mean'], df_data['TT']))
	axs[0].set_ylabel('Cell-wise mean')
	hide_spines()
	plt.tight_layout()

	# Attempt distribution decoding from these neurons
	# Determine optimism/pessimism by comparing to t-distribution

	var_types = protocol_info['var_types']
	t_fig, t_axs = plt.subplots(periods['n_periods_to_plot'], len(var_types), figsize=(12, 6))

	low_tt = protocol_info['low_tt']
	high_tt = protocol_info['high_tt']
	low_resp = X_tt_means[:, np.newaxis, :, low_tt]  # n_cells x 1 x n_periods
	high_resp = X_tt_means[:, np.newaxis, :, high_tt]  # n_cells x 1 x n_periods
	# low_resp = X_tt_means[:, :, low_tt]  # n_cells x n_periods
	# high_resp = X_tt_means[:, :, high_tt]  # n_cells x n_periods

	# Dabney et al: "We  first  normalized  the  responses  to the 50% cue on a per-cell basis as follows,
	# [see paper], where mean indicates the mean over trials within a cell"

	# Find the actual minimal and maximal average response over all trial types, rather than assume which type it is
	# low_resp = np.expand_dims(np.amin(np.array([np.mean(X_means[:, trial_inds_inc_types[tt], :], axis=1) for tt in range(n_active_types)]), axis=0), 1)
	# high_resp = np.expand_dims(np.amax(np.array([np.mean(X_means[:, trial_inds_inc_types[tt], :], axis=1) for tt in range(n_active_types)]), axis=0), 1)
	diff_resp = high_resp - low_resp

	t_stats = []
	for i, i_type in enumerate(var_types):
		# following Dabney et al., 2020, normalize the response (separately for all trials?)
		tt_activities_i = X_means[:, trial_inds_inc_types[i_type], :]  # n_cells x n_trials_of_type_i x n_periods
		tt_activities_norm_i = (tt_activities_i - low_resp) / diff_resp
		# average across trials to get the mean normalized response for each cell
		cell_means_norm_i = np.mean(tt_activities_norm_i, axis=1)  # n_cells x n_periods
		tt_activities_norm_i_list = [tt_activities_norm_i[i_cell, ...] for i_cell in range(n_cells)]

		# old way: average over trials before normalizing
		#     cell_means_i = np.mean(X_means[:, trial_inds_inc_types[i_type], :], axis=1)  # n_cells x n_periods
		#     cell_means_norm_i = (cell_means_i - low_resp) / diff_resp  # n_cells x n_periods

		out = plot_t(i, cell_means_norm_i, tt_activities_norm_i_list, n_cells, periods, protocol_info, t_axs)
		#     est_taus, _, _ = t_decode(i, cell_means_norm_i, est_taus, [[0], [0]], n_cells, periods, protocol_info, t_axs)
		t_stats.append(out)

	hide_spines()
	t_fig.tight_layout()
	t_fig.savefig('t_fig.svg', bbox_inches='tight')

	# c: n_probs x n_cells x max_n_trials array of cue responses
	mpl_params = {'font.size': 12}
	mpl.rcParams.update(mpl_params)

	method = 'mse'
	name = '_'.join([protocol, 'D1_rew_inds', method, mouse_name, file_date])
	savepath = os.path.join(name, name + '.p')
	if os.path.exists(savepath):
		pop = dill.load(open(savepath, 'rb'))
	else:
		if protocol == 'Bernoulli':
			pop = Bernoulli(p=np.array([.01, .2, .5, .8, .99]), c=cue_resps, name=name, method=method)
			pop.regress()
			pop.fit_param_expectile_cv()
		else:
			var_norm_dists = [protocol_info['norm_dists'][i] for i in var_types]
			cue_resps_to_use = cue_resps[var_types]
			pop = PopCode(var_norm_dists, cue_resps_to_use, name=name, method=method)
		pop.fit_dabney_expectile_cv()
		taus = pop.dabney_xopts[:, -1]
		dill.dump(pop, open(savepath, 'wb'))

	if protocol == 'Bernoulli':
		pop.plot_all_cells()
		pop.plot_collapsed()
	else:
		fig, axs = plt.subplots(1, 2, figsize=(6, 3))
		pred = np.zeros((pop.n_dists, pop.n_cells))
		handles = [[]] * pop.n_dists
		pop.colors = plt.cm.copper(np.linspace(0, 1, pop.n_dists))
		for i_cell in range(pop.n_cells):
			pred[:, i_cell] = pop.predict(pop.dabney_xopts[i_cell])
			pop.dabney_exp_all_pred[:, i_cell] = pred[:, i_cell]
		for ip in range(pop.n_dists):
			handles[ip] = axs[0].scatter(pred[ip], pop.c_test_avg[ip, :], marker='o', color=pop.colors[ip])
		pop.plot_correl(pred, pop.c_test_avg, axs[0], 'Dabney')

		axs[-1].legend(handles=handles, labels=protocol_info['trial_type_names'][1:4], loc='upper left')
		axs[-1].axis('off')
		#     fname = os.path.join(self.name, '_'.join((self.name, 'collapsed.png')))
		plt.tight_layout()
	#     plt.savefig(fname, dpi=300, bbox_inches='tight')

	dec_fig, dec_axs = plt.subplots(periods['n_periods_to_plot'], len(var_types), figsize=(12, 6))
	# imp_dists = plot_decode(t_stats, periods, protocol_info, dec_axs, taus)
	imp_dists = plot_decode(t_stats, periods, protocol_info, dec_axs)
	hide_spines()
	dec_fig.tight_layout()

	plot_pairs(t_stats, periods, var_types, 'x_norm')

	# CHANGE THIS VARIABLE TO PLOT DIFFERENT CELLS
	# i_cell = 2
	# fig, axs = plt.subplots(2, len(var_types), figsize=(10, 4))

	# alternatively, plot all cells
	fig, axs = plt.subplots(n_cells + 1, len(var_types), figsize=(10, 3 * n_cells))

	# compare variable cells to high and low reward
	# low_tt = 4; high_tt = 5
	low_timecourse = cell_means_tt[..., low_tt]  # n_cells x n_timepoints
	high_timecourse = cell_means_tt[..., high_tt]  # n_cells x n_timepoints

	for i, i_type in enumerate(var_types):

		i_timecourse = cell_means_tt[..., i_type]  # n_cells x n_timepoints
		i_colors = np.vstack((colors['colors'][low_tt], colors['colors'][i_type], colors['colors'][high_tt]))

		for i_cell in range(n_cells):
			to_plot = np.vstack((low_timecourse[i_cell], i_timecourse[i_cell], high_timecourse[i_cell])).T
			ax = axs[i_cell, i]
			#         ax = axs[0, i]
			ax.set_prop_cycle('color', i_colors)
			h = ax.plot(timestamps['time'], to_plot)
			ax.axvspan(0, timestamps['stimulus_dur'], color=colors['vline_color'], alpha=0.3)
			ax.axvline(timestamps['stimulus_dur'] + timestamps['trace_dur'], color=colors['vline_color'], alpha=0.3)
			if i == 0:
				ax.set_ylabel(activity_label)
			if i_cell == n_cells - 1:
				ax.set_xlabel('Time from CS')

		ax = axs[n_cells, i]
		#     ax = axs[1, i]
		ax.legend(handles=h, labels=['CS {}'.format(tt) for tt in [low_tt, i_type, high_tt]], loc='upper left')
		ax.axis('off')

	hide_spines()
	fig.tight_layout()

	# compare population variance across CSs. Each dot is a trial, so there are different numbers in each group
	# and samples are unpaired
	fig, axs = plt.subplots(1, periods['n_periods_to_plot'], figsize=(4 * periods['n_periods_to_plot'], 3))
	tt_concat = np.concatenate([[i] * len(subl) for (i, subl) in enumerate(trial_inds_inc_types)])
	for i, i_period in enumerate(periods['periods_to_plot']):
		pop_vars = []
		ax = axs[i]
		for i_type in active_types:
			tt_means_i = X_means[:, trial_inds_inc_types[i_type], i_period]
			pop_vars.append(np.var(tt_means_i, axis=0))  # variance across cells
		sns.swarmplot(data=pop_vars, ax=ax, palette=colors['palette'])
		ax.set_title(periods['period_names'][i_period])
		if i == 0:
			ax.set_ylabel('Population Variance')
		ax.set_xlabel('CS')
		ax.set_xticklabels(active_types)
		print(periods['period_names'][i_period])
		print(kruskal(*pop_vars))
		print(pairwise_tukeyhsd(np.concatenate(pop_vars).ravel(), tt_concat))

	hide_spines()

	plot_cs_cs_corr(X_tt_means, periods, var_types)

	# PCA/NMF across neural dimension
	n_comps = 3
	pca = PCA(n_components=n_comps)
	print(X_concat.T.shape)
	X_neu = pca.fit_transform(X_concat.T)
	fig, axs = plt.subplots(1, 2, figsize=(14, 4))
	axs[0].plot(X_concat_time, X_neu)
	axs[0].set_xlabel('Time (s)')
	axs[0].set_title('PCs')

	print(X_concat.min())
	print(X_concat.max())
	nmf = NMF(n_components=n_comps)
	X_neu_nmf = nmf.fit_transform(X_concat.T - X_concat.min())
	axs[1].plot(X_concat_time, X_neu_nmf)

	# PCA across temporal dimension, then hierarchcial clustering on PCA timecourses
	pca = PCA(n_components=n_comps)
	print(X_concat.shape)
	X_tem = pca.fit_transform(X_concat)
	print(X_tem.shape)

	fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [6, 1, 1], 'wspace': 0.1}, figsize=(20, 6))
	Z = linkage(X_tem, metric='euclidean', optimal_ordering=True)
	# leaves run from bottom to top here
	R = dendrogram(Z, ax=axs[2], orientation='right')  # leaf_label_func=llf, leaf_rotation=60.,leaf_font_size=12.)
	all_axs_off(axs[2])

	print(R['leaves'][::-1])
	X_tem_sort = X_tem[R['leaves'][::-1], :]
	axs[1].pcolormesh(np.arange(.5, n_comps + 1), np.arange(n_cells + 1), X_tem_sort)
	axs[1].set_yticks([])  # now it should be sorted correctly

	prc_range = np.percentile(X_concat, [2.5, 97.5], axis=None)
	this_cmap = cmocean.tools.crop(cmocean.cm.balance, prc_range[0], prc_range[1], pivot=0)
	im = axs[0].pcolormesh(X_concat_pcolor_time, np.arange(n_cells + 1), X_concat[R['leaves'][::-1], :],
	                       vmin=prc_range[0], vmax=prc_range[1], cmap=this_cmap)
	add_cbar(fig, im, 'fluorescence (std)')
	axs[0].set_xlabel('Time (s)')
	axs[0].set_yticks([])
	axs[0].set_ylabel('Neuron')

	# hierarchical clustering on timecourses
	# fig, axs = plt.subplots(2, 2, figsize=(20, 12), gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.02})
	fig = plt.figure(figsize=(20, 12))
	grid = plt.GridSpec(2, 2, height_ratios=[1, 2], hspace=0.02)
	Z = linkage(X_concat, metric='correlation', optimal_ordering=True)
	dend_ax = fig.add_subplot(grid[0, 1])
	R = dendrogram(Z, ax=dend_ax)
	all_axs_off(dend_ax)

	# take the correlation of every cell with every other cell, and see what it looks like
	X_ordered = X_concat[R['leaves']]
	C = np.corrcoef(X_ordered, rowvar=True)
	C_ax = fig.add_subplot(grid[1, 1])
	im = C_ax.pcolormesh(range(n_cells + 1), range(n_cells + 1), C, vmin=-1, vmax=1, cmap=cmocean.cm.balance)
	C_ax.set_ylim(n_cells, 0)
	C_ax.set_yticks([])
	C_ax.set_xticks([])
	add_cbar(fig, im, 'correlation')

	# just plot the raw (concatenated) flurorescence transients
	raw_ax = fig.add_subplot(grid[:, 0])
	raw_ax.plot(np.arange(n_trials * n_samps_per_trial), X_ordered.T + np.arange(n_cells) * 10, lw=.5)
	all_axs_off(raw_ax)

	# plot concatenated fluo traces with stimulus/reward delivery marked
	fig, axs = plt.subplots(figsize=(35, 10))
	plt.plot(np.arange(n_trials * n_samps_per_trial), X_ordered.T + np.arange(n_cells) * 10, lw=.5)
	plt.vlines(np.arange(odor_on_idx, n_trials * n_samps_per_trial, n_samps_per_trial), 0, n_cells * 10, lw=.5)
	plt.vlines(np.arange(trace_end_idx, n_trials * n_samps_per_trial, n_samps_per_trial), 0, n_cells * 10, 'r', lw=.5)
	all_axs_off(axs)

	# perform spectral clustering on tuning vector
	tuning_vector = np.reshape(X_tt_means, (n_cells, -1))
	spectral = SpectralClustering(n_clusters=5)  # , affinity='nearest_neighbors')
	spectral_labels = spectral.fit_predict(tuning_vector)
	X_spect = X_concat[np.argsort(spectral_labels)]

	fig, axs = plt.subplots(1, 2, figsize=(20, 8))
	axs[0].plot(np.arange(n_trials * n_samps_per_trial), X_spect.T + np.arange(n_cells) * 10, lw=.5)
	all_axs_off(axs[0])

	ax = axs[1]
	# C = np.corrcoef(X_spect, rowvar=True)
	# im = ax.pcolormesh(range(n_cells+1), range(n_cells+1), C, vmin=-1, vmax=1, cmap=cmocean.cm.balance)
	im = ax.pcolormesh(range(n_cells + 1), range(n_cells + 1), spectral.affinity_matrix_)
	ax.set_ylim(n_cells, 0)
	ax.set_yticks([])
	ax.set_xticks([])
	add_cbar(fig, im, 'affinity')

	# Representational Similarity Analysis (RSA)
	plot_RSA(X_tt_means, n_active_types)

	# perform classification of CS types using many different classifiers, for comparison
	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
	         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
	         "Naive Bayes", "Logistic Regression"]

	classifiers = [
		KNeighborsClassifier(3),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		GaussianProcessClassifier(1.0 * RBF(1.0)),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		MLPClassifier(alpha=1, max_iter=1000),
		AdaBoostClassifier(),
		GaussianNB(),
		LogisticRegression(solver='newton-cg', multi_class='multinomial')]

	# X = StandardScaler().fit_transform(X)
	# iterate over classifiers, performing K-fold cross-validation
	trace_results = []
	scoring = 'balanced_accuracy'
	kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
	for name, clf in zip(names, classifiers):
		cv_trace_results = cross_val_score(clf, X_means[..., main_comp_ind].T, trial_inds, cv=kfold, scoring=scoring)
		trace_results.append(cv_trace_results)
		print("%s: %f (%f)" % (name, cv_trace_results.mean(), cv_trace_results.std()))

	# boxplot algorithm comparison
	plt.figure(figsize=(10, 8))
	plt.title('Algorithm Comparison')
	plt.boxplot(trace_results)
	plt.axhline(1. / n_active_types, ls='--', c=[.4, .4, .4])
	plt.xticks(ticks=range(len(names) + 1), labels=[''] + names, fontsize=8)

	# regress reward quantity delivered on reward response
	# first, examine correlation of individual cells with reward amount
	fig, axs = plt.subplots(int(np.ceil(np.sqrt(n_cells))), int(np.ceil(np.sqrt(n_cells))))
	rs = []
	rew_period = 3
	for i in range(n_cells):
		ax = axs.flat[i]
		ax.scatter(X_means[i, :, rew_period], reward)
		rs.append(np.corrcoef(X_means[i, :, rew_period], reward)[0, 1])
	hide_spines()
	fig.tight_layout()

	plt.figure()
	plt.hist(rs)

	# GLM of neural activity (during reward epoch) on reward size. Use elastic net regularization
	param_grid = [{'l1_ratio': [1e-5, 0.01, 0.05, .1, .3, .5, .7, .9, 1]}]
	cv = KFold(n_splits=4, shuffle=True, random_state=1)
	enet = GridSearchCV(ElasticNetCV(cv=cv), param_grid, cv=cv, iid=False)
	enet.fit(X_means[..., rew_period].T, reward)
	print(enet.best_params_)
	cv_rew_results = cross_val_score(enet, X_means[..., rew_period].T, reward, cv=cv)
	print("%s: %f (%f)" % ('Linear Regression R^2', cv_rew_results.mean(), cv_rew_results.std()))

	embedding = Isomap(n_components=2)
	X_iso = embedding.fit_transform(X_concat.T)

	# color based on time in trial
	plt.scatter(X_iso[:, 0], X_iso[:, 1], c=time_concat, alpha=.5)
	plt.colorbar()

	# color based on period of trial
	plt.figure()
	plt.scatter(X_iso[:, 0], X_iso[:, 1], c=epoch_concat, alpha=.5)

	# color based on trail type
	plt.figure()
	plt.scatter(X_iso[:, 0], X_iso[:, 1], c=trial_inds_iso, alpha=.5, cmap='copper')
	plt.colorbar()