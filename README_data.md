# Data from: An opponent striatal circuit for distributional reinforcement learning

[https://doi.org/10.5061/dryad.80gb5mm0m](https://doi.org/10.5061/dryad.80gb5mm0m)

## Description of the data and file structure

We trained thirsty mice in a classical conditioning task in which random odors were paired with different probability distributions over water amounts. Behavior was monitored using an IR lick sensor, a face camera, and a body camera, and non-licking behavioral variables were extracted using Facemap. We recorded from neurons in the anterior striatum using acute Neuropixels probes and registered each cell to the Allen Common Coordinate Framework. In a subset of mice, we injected 6-hydroxydopamine unilaterally into the lateral nucleus accumbens shell (lAcbSh) before training to lesion dopamine axons projecting to this region, and then recorded after training. Separately, we recorded specifically from lAcbSh neurons expressing either the D1 or D2 dopamine receptor using two-photon calcium imaging in transgenic mice. Finally, we expressed excitatory (CoChR) or inhibitory (GtACR1) opsins in these cell types and measured changes in licking induced by optogenetic stimulation.

## Introduction 

This `data` folder is meant to be located at the same level of the folder hierarchy as `code`. Files should then be found automatically by the `code` scripts, hosted by Zenodo: [https://doi.org/10.5281/zenodo.14172350](https://doi.org/10.5281/zenodo.14172350). Here is an example:

```
├── parent_dir
│   ├── code
│   ├── ├── neural_analysis
│   ├── ├── behavior_analysis
│   ├── ├── ann_decoding
│   ├── ├── envs
│   ├── ├── ...
│   ├── data
│   ├── ├── neural-plots
│   ├── ├── behavior-plots
│   ├── ├── ann_decoding
│   ├── ├── behavior
│   ├── ├── ...
```

`data` contains nine (individually zipped) subfolders: 'neural-plots', 'behavior-plots', 'ann_decoding', 'behavior', 'camera', 'ephys', 'fano', 'glm', and 'imaging'. These will be described in turn. 'neural-plots' is the most important and is used to generate the majority of the figures in the paper.

In addition to these nine subfolders, there are three files: this README.md, session_log.zip (to be unzipped into the SQLITE database file session_log.sqlite), and session_log_config.json, which contains the schema for the database. This database contains information about all mice, behavior sessions, ephys/imaging sessions, etc., saved as separate tables. This will be queried frequently to pull up the relevant sessions for analysis.

Files that end in '.p' can be loaded in Python using `pickle`. Those that end in '.sav' can be loaded using `joblib`.

Several abbreviations are commonly used in many file names. These include the following:

#### Tasks (often called "protocols")
- 'SameRewDist' (the Same Reward Distribution task, the primary task described in the main text of the paper)
- 'Bernoulli' (Bernoulli task, Fig. 8)
- 'DiverseDists' (Diverse Distributions task, Fig. 8)
- 'SameRewVar' (Fourth Moments task, Fig. 8)

#### Recording modalities
- 'ephys' (electrophysiology)
- 'imaging' (two-photon calcium imaging)

#### Activity types
- 'spks': raw spike counts (ephys) or deconvolved Ca2+ activity (imaging)
- 'firing': smoothed firing rate traces
- 'zF': z-scored fluorescence (imaging only)

#### Manipulation
- 'combined' (no manipulation, or combining neurons from fully intact animals and the control hemisphere of lesioned animals. In the case of 'imaging', this combines both D1 and A2a-Cre animals, which will be labeled and disaggregated elsewhere)
- '6-OHDA' (contrasts control vs. lesioned hemispheres in the 6-OHDA dopamine lesion experiments)

### Useful variable names

This README.md document will make frequent use of the following variable names, which are described here:

- `n_trace_types`: the number of trial types that include a Trace period. This is equal to the number of odor types.
- `n_trial_types`: `n_trace_types + 1`, since this includes the Unexpected Reward trial type


## `neural-plots`


This folder contains files with ephys and imaging data for the various protocols, mostly used by `neural_analysis/recording_figs.ipynb`. There are three types of files.

Files ending with 'spks.sav' (e.g. `SameRewDist_ephys_combined_striatum_spks.sav`) are the most important. They contain a `dict` with the following keys:

- `all_spk_cnts`: an array of spike counts binned every 250 ms, shape (n_trial_types, total_cells, max_n_trials_per_type, n_psth_bins). max_n_trials_per_type was set to 90; unused trials are np.nan. n_trial_types = n_trace_types + 1, as it includes  Unexpected Reward trials.
- `cue_resps`: an array of spike counts binned every 1 s, shape (n_trace_types, total_cells, max_n_trials_per_type, n_prerew_periods). n_prerew_periods = n_periods - 1 = 4, as it excludes Outcome.
- `X_means`: an array of trial-average firing rates, shape (n_trial_types, total_cells, n_periods)
- `cell_stds`: standard deviation of each cell's firing rate, shape (n_trial_types, total_cells, n_periods)
- `neuron_info`: dictionary (for pickling purposes) which can and should be restructured as a DataFrame using pd.DataFrame(neuron_info) with shape (total_cells, 23). Contains information like mouse, coordinates, mean, std, `class_name`, and striatal subregion (str_regions) for each neuron in `all_spk_cnts`, `cue_resps`, etc. in order. See documentation for `glm` folder for more information.
- `late_trace_ind`: equal to 3, which period index (e.g. of `X_means` or `cue_resps`) corresponds to the Late Trace period
- `n_trace_types`: protocol-specific, usually 6 (for 6 odors used), but sometimes 5 (e.g. Bernoulli task), for the number of odors with trace periods
- `psth_bin_width`: 250 ms, bin width for PSTH
- `psth_bin_centers`: centers of PSTH bins
- `corrs`: correlations with e.g. mean, reward, rpe, etc, computed at psth_bin_width with psth_bin_centers
- `corrs_seconds`: correlations with e.g. mean, reward, rpe, etc, computed during each period (seconds)
- `n_psth_bins`: len(psth_bin_centers)
- `n_periods`: equal to 5, the number of periods used, each 1 s long (Baseline, Odor, Early Trace, Late Trace, Outcome)
- `total_cells`: number of cells, len(neuron_info)

Second, and also important, are files containing 'dec_dict'. These contain data pertaining to the decoding analyses: CCGP, pairwise decoding, congruency, mean, and odor. Each file is a (highly nested) dictionary with two keys, `per` and `bin`. `per` is binned at (4) 1 second periods (Baseline, Odor, Early Trace, Late Trace), while `bin` is binned at (24) 250 ms periods, spanning from 1 s prior to odor delivery to 2 s after reward delivery. Call the length of these `n_periods` (either 4 or 24). While these data are included, it is also possible to regenerate them (perhaps with different random seeds) using the data above and the included code.

Each of these contains their own nested dictionary with keys `ccgp`, `pair`, `cong`, `mean`, and `odor`, for the various types of decoding analyses described in the paper. These, then contain their own dictionary with the following fields:

- `name`: the name of the decoding analysis, e.g. 'ccgp'
- `keys`: the name of each grouping, in order, e.g. 'Distribution CCGP 1' through 4 in the case of CCGP; 'Congruent' and 'Incongruent 1' through 2 for congruency analysis
- `colors`: hex codes for each key
- `resps`: dict with keys given by `keys`. For CCGP, the shape of each value is `(2, total_cells, max_n_trials_per_type, 2, n_periods)`. The zero-th dimension splits the trials assigned to each class for training. The third dimension splits the trials assigned to each class for testing. For all decoders other than CCGP, the same trial types (but different trials) are used for training and testing. Therefore, the shape of the relevant arrays is `(2, total_cells, max_n_trials_per_type, n_periods)`, with the zero-th dimension containing the trial types assigned to each of the two classes. Odor decoding is six-way, so the zero-th dimension has size 6.
- `within_dist_keys`: list containing all keys from across the different decoders that are considered "Within distribution" comparisons: ['Fixed 1 vs. Fixed 2', 'Variable 1 vs. Variable 2', 'Incongruent 1', 'Incongruent 2', 'Fixed vs. Variable', 'Uniform 1 vs. Uniform 2', 'Bimodal 1 vs. Bimodal 2', 'Uniform vs. Bimodal']
- `pooled_colors`: list containing colors in which to plot 'pooled' estimates (average of all Across- or Within-distribution pairs), e.g. ['#74B72E'] (for CCGP, in which there is no "within distribution" equivalent) or ['#74B72E', '#FF6600'] (for congruency and pairwise)
- `pooled_keys`:  list containing keys that apply to 'pooled' estimates/colors e.g. ['Across distribution', 'Within distribution'] (for congruency and pairwise)
- `train_pers`: which periods to train on, in order (e.g. `np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])`)
- `test_pers`: which periods to test on, in order (e.g. `np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])`). This arrangement is useful for cross-temporal decoding
- `scores`: this is a highly nested dictionary. At the top level there are two keys, 'all' and 'shuff'. The vast majority of analyses just use 'all'. 'shuff' assigns a new, random trial type label to each trial for each neuron independently (in a manner that preserves the total number of trials for each trial type). In other words, we randomly permute the trial type labels across all trials, and repeat this procedure separately for every neuron.
- - Within this is a dictionary where the keys are subregions (or 'All Subregions').
- - - Within this dictionary, the keys refer to individual mice (e.g. 'AL39'). 
- - - - Within this are individual sessions for simultaneous decoding, labeled by their `figure_path`. (We only used simultaneous decoding in the case of 'All Subregions' for control mice, i.e. Fig. 2)
- - - - Otherwise, it simply contains 'pseudo'. This is the case for individual subregions (for which we pooled across sessions within animals), as well as 6-OHDA lesions
- - - - - Within this dictionary we have another dictionary with keys `pop_size` (the number of cells used for decoding, an integer) and the `keys` above that are unique to each decoder.
- - - - - - Within these dictionaries are the `period_keys`, e.g. '3_3' for training and testing on the Late Trace period.
- - - - - - - Finally, we come to a list, returned by either `simultaneous_decode()` or `disjoint_decode()` within `neural_analysis/analysisUtils.py`.
- - - - - - - For simultaneous decoding, the order is `[score, coefs, confusion, correct]`. `score` is the decoder accuracy on the test set; `coefs` is an array of shape `(n_cv_splits, n_pairs, n_cells)` containing the learned classifier weights for each neuron in the (linear) classifier in each CV fold (`n_pairs = 1` except for odor decoding); `confusion` is the confusion matrix for odor decoding (the only time it isn't binary classification); and `correct` is a Boolean array of shape `(n_trials,)` indicating whether or not that trial was classified correctly (except for CCGP, where it is `None`).
- - - - - - - For pseudopopulation decoding, the order is `[pop_scores, coefs, cell_inds_all, confusions]`. `pop_scores` is an array of shape `(n_pseudopop_sizes, n_cv_splits)`, with `pseudopop_sizes` in increasing order; `coefs` is an array of shape `(n_pseudopop_sizes, n_cv_splits, n_pairs, n_cells)` containing the learned classifier weights for each neuron in the (linear) classifier in each CV fold, as above; `cell_inds_all` is an object array of shape `(n_pseudopop_sizes,)`, where each element is itself a Boolean array of length `n_cells` indicating whether or not a given cell was included in that (randomly-sampled) pseudopopulation; and `confusions` is an array of shape `(n_pseudopop_sizes, n_cv_splits, n_trace_types, n_trace_types)` containing the confusion matrices for each pseudopopulation and CV split.


Third, files ending with 'spks_smooth' or 'firing' contain the smoothed data necessary for plotting (in `neural_analysis/plot_smoothed_data.ipynb`). This is also a `dict` with the following fields:

- `neuron_info`: as above
- `n_trace_types`: as above
- `high_tt_concat`: empty; ignore
- `pcolor_time_full`: array of shape (6001,), containing the timebase for pseudocolor plots
- `timestamps`: a dictionary with the keys:
- - `foreperiod`: int, duration of Baseline period in seconds (1)
- - `iti`: int, duration of minimum ITI period in seconds (2)
- - `align`: array of shape (n_trials,) containing the time in seconds on which to align each trial, equal to the time of odor onset (or reward onset for Unexpected Reward trials)
- - `trial_start`: array of shape (n_trials,) containing the absolute time in seconds at which each trial began (from hitting "Run")
- - `ttl_high`: array of shape (n_trials,) containing the absolute time in seconds at which the TTL from the camera went high (used for sync; beginning of foreperiod)
- - `ttl_low`: array of shape (n_trials,) containing the absolute time in seconds at which the TTL from the camera went low (used for sync; beginning of ITI)
- - `trace`: float, trace duration in seconds, (2.0)
- - `stim`: float, odor duration in seconds, (1.0)
- - `ttls`: array of shape (n_trials * 2,) containing the absolute time in seconds of all TTLs, high or low
- - `bin`: float, bin width for licks in seconds (0.001)
- - `trace_trial`: array of shape (n_trials,) containing the trace duration on each trial (2.0 unless Unexpected Reward, in which case 0.0)
- - `stim_trial`:  array of shape (n_trials,) containing the stimulus duration on each trial (1.0 unless Unexpected Reward, in which case 0.0)
- - `time`: array of shape (6000,), containing `pcolor_time_full[:-1]`
- `timecourses`: array of shape `(n_trial_types, total_cells, 6000)`, containing the average response of each neuron to each trial type across `time`. In this case the averaging has been performed on smoothed timecourses.

Finally, in addition to these files, there are three subfolders corresponding to the three subjects whose data are specifically plotted in `neural_analysis/plot_sample_data.ipynb` (e.g. `AL60`). These subfolders start out either empty or with several .png files; they will be fully populated after running the code in this notebook. These subfolders are not created automatically in the code, so it will throw an error if they are deleted.

## `behavior-plots`

This folder contains processed data from the example mice/sessions used in Fig. 1h and 4d,g. It also contains several combined data files used for facemap decoding (e.g. Fig. 1d) and quantification of optogenetic effects (Fig. 5).

The file 'SameRewDist_facemap_decoding_104_9853e89f4e8252dc9d707f95c8909d05.p' is used in `behavior_analysis/behavior_decoding.ipynb`. It contains the following fields:

- `beh_resps`: array of shape `(n_trace_types, n_predictors, max_n_trials_per_type, n_periods)` containing the average predictor (e.g. facemap component) during each 1 s period of each trial
- `bin_resps`: array of shape `(n_trace_types, n_predictors, max_n_trials_per_type, n_bins)`, as above but containing the average predictor (e.g. facemap component) during each 250 ms bin of each trial
- `rewards`: array of shape `(n_trace_types, n_predictors, max_n_trials_per_type)` containing the amount of reward delivered in each trial
- `beh_info`: pandas DataFrame of shape `(n_predictors, 4)` containing these columns:
- - `names`: mouse name
- - `file_date_id`: recording date
- - `fig_paths`: path to file, not operable but can be used to uniquely identify session
- - `class_name`: either 'helper' (for standard condition) or 'lesion' (for 6-OHDA experiment, identifying 'control' or 'lesioned' mice)
- `ret_df`: pandas DataFrame of shape `(n_sessions, 42)` containing relevant entries from `session` table of database.

The files 'SameRewDist_dec_dict_facemap_helper_6_0.005.p' and 'SameRewDist_dec_dict_facemap_lesion_6_0.005.p' are structured exactly as the dec_dict files in the `neural-plots` folder.

Lastly, the files beginning with 'compare_opto' contain information from optogenetic stimulation sessions (used in `behavior_analysis/compare_optostim.ipynb`). The one ending in '2.5.p' uses the lick rate in just the last half-second of the Trace period, as in the paper. The one ending in '2.0.p' uses the entire last second of the Trace period ("Late Trace"). The results are qualitatively identical. They each contain a pandas DataFrame in long format of shape `(57000, 13)`. Each row is a unique trial. Columns are as follows:

- `name`: mouse name
- `genotype`: mouse genotype
- `excitation`: Boolean, whether it was excitation (True) or inhibition (False)
- `exp_date`: experiment date
- `trial_type`: int, indicating Nothing (0), Fixed (1) or Variable (2)
- `stim_trial`: Boolean, indicating whether there was stimulation on that trial
- `stim_loc`: int, indicating stimulation location [0, 1, 2, 3]
- `loc_labels`: string, indicting stimulation location [No Stimulation, Ventral, Intermediate, Dorsal], can be indexed by `stim_loc`
- `licks`: lick rate during that trial
- `trial_num`: trial number during session
- `next_licks`: lick rate during the subsequent trial (np.nan if last trial)
- `next_stim_loc`: `stim_loc` during the subsequent trial
- `next_trial_type`: trial type of subsequent trial

Similar to `neural-plots`, there are also three subfolders corresponding to the three subjects whose data are specifically plotted in `neural_analysis/plot_sample_data.ipynb` (e.g. `AL60`). These subfolders (and their session-specific sub-subfolders) contain a single pickle file with processed behavioral data from that session. This file contains a dictionary with the following keys:

- `stats`: a dictionary containing the results of statistical tests performed on that entire session's data
- `licks_smoothed`: an array of shape `(n_trials, 35000)` containing smoothed licking traces on each trial, after binning at 1 ms resolution and spanning from -6 to 30 s, aligned to odor onset at 0 s. Traces were convolved with a Gaussian window with SD = 5 ms.
- `licks_raw`: an array of shape `(n_trials, 35000)` containing unsmoothed lick counts on each trial, after binning at 1 ms resolution and spanning from -6 to 30 s, aligned to odor onset at 0 s.
- `active_types`: list containing trial types that appeared during that session (e.g. [0, 1, 2, 3, 4, 5, 6], where 6 corresponds to Unexpected Reward)
- `trial_types`: int array of shape `(n_trials,)` containing trial type for each trial in order
- `time`: array of shape `(35000,)`, giving the timebase for `licks_raw` and `licks_smoothed`
- `sr`: sampling rate in Hz, e.g. 1000 (corresponding to ms resolution)

## `ann_decoding`

The folders 'decoding_SameRewDist_pseudo', 'decoding_SameRewDist_pseudo_restrictedSameMean', and 'decoding_SameRewDist_pseudo_transfer' contain the results of running the relevant scripts contained in the 'code/ann_decoding' folder. These scripts depend on the file 'SameRewDist_combined_spks_data_20230918.p', also contained in this folder. This contains a dictionary with the following fields:

- `protocol_info`: a dictionary containing information about the task under consideration, including the distributions used for each trial type (`dists`), and the `mean`, variance (`var`), standard deviation (`std`), and conditional value at risk (`cvar`) of each distribution.
- `cue_spk_cnts`:  an array of shape `(n_trace_types, total_cells, max_n_trials_per_type, n_periods)` containing spike counts in each 1 s period.
- `neuron_info`: as above, but with additional columns like `cutoff_73`, a Boolean indicating whether each neuron was above the 73rd percentile cutoff
- `dec_dict`: as above, but only for `per`, not `bin`. Also, we include two additional keys, 'coef_arr' and 'diff_coef_arr'. 
- - `coef_arr`: if 'ccgp', then this is an array of shape `(total_cells, n_groupings, 2)`, with `n_groupings = 4` for Distribution CCGP 1-4.  The last dimension compares either ordered or shuffled coefficients. If not 'ccgp', then this is an array of shape `(total_cells, n_groupings, n_cv_folds, 2)`, since now there is no completely held-out test set and we must train `n_cv_folds` different classifiers.
- - `diff_coef_arr`: an array of shape `(total_cells, 2)`. For 'ccgp', this is just the average over gropuings. Otherwise, it is the difference in the average of Across $-$ Within distribution groupings, averaged again over CV folds.
- `cutoff`: an array of the cutoffs tested and put into `neuron_info`. 73 was used. 

## `behavior`

This folder contains two types of files.

Those that begin with 'licking' (e.g. 'licking_SameRewDist_ephys.p') summarize the licking behavior for the cohort of animals trained in that task and recorded using that image modality (although it may include additional sessions during which neural recording did not take place). It contains the following fields:

- `last_sec_means`, `half_sec_means`, `trace_lick_means`, `baseline_lick_means`, `rew_means`: arrays of shape `(n_trace_types, n_sess)`, containing the average number of licks during the last second prior to reward (Late Trace period), last half-second prior to reward, the full Trace period (2-0 s before reward), Baseline period (1-0 s before odor), and Outcome (0-1 s after reward) periods, respectively.
- `protocol_info`: a dictionary containing information about the task under consideration, including the distributions used for each trial type (`dists`), and the `mean`, variance (`var`), standard deviation (`std`), and conditional value at risk (`cvar`) of each distribution.
- `n_sess`: number of sessions

The second type of file begins with 'var_licking' (e.g. 'var_licking_SameRewDist_ephys.p'). This contains information about licking following Variable trials, with the following fields:
- `var_means`: array of shape `(n_var_rews, n_var_tts, n_sess)`, containing the average number of licks  during the Late Trace period in each session. The zero-th dimension indexes the previous trial's reward (2 or 6), while the first dimension indexes the next trial's reward (2 or 6 uL).
- `var_name`: name of Variable trial types
- `n_var_rews`: number of Variable rewards (e.g. 2)
- `n_var_tts`: number of Variable trial types (e.g. 2)
- `var_inds`: which trial/trace types are variable (0-indexed, e.g. [4, 5])

Also included in this folder are several subfolders, for mouse subjects 'AL39', 'AL60', and 'AL65'. At the bottom of each of these directory trees is raw data from a single recording session (e.g. 'AL39_SameRewDist_20210930_111035.mat'), used in `neural_analysis/plot_sample_cells.ipynb`. This is not important for understanding the analysis here. For detailed description, see the Bpod documentation. 

## `camera`

This folder contains data from a single example session, mouse 'AL41' on day '20211102', hence 'AL41_20211102_proc.npy'. It is used in `behavior_analysis/plot_facemap_components.ipynb` and contains the output of facemap processing. It is a numpy object array with the following important fields, copied from the [facemap documentation](https://facemap.readthedocs.io/en/stable/outputs.html). Details for others can be obtained from the facemap documentation.

- `pupil`: list of pupil ROI outputs - each is a dict with ‘area’, ‘area_smooth’, and ‘com’ (center-of-mass)
- `running`:  list of running ROI outputs - each is nframes x 2, for X and Y motion on each frame
- `motion`: list of absolute motion energies across time - first is “multivideo” motion energy (empty if not computed). This is what gets called 'whisking'
- `motSVD`:  list of motion SVDs - first is “multivideo SVD” (empty if not computed) - each is of size number of frames by number of components (50)
- `running`: list of running ROI outputs - each is nframes x 2, for X and Y motion on each frame

## `ephys`

This contains a directory tree with structure `mouse_name/filedate/channel_locations.json`. Each `channel_locations.json` file is the output from the IBL atlas-electrophysiology pipeline, after registering the depth of the probe insertion to the allen CCF using electrophysiological coordinates.  It contains x (ML), y (AP) and z (DV) coordinates in micrometers relative to bregma, along with the allen CCF brain region IDs of this coordinate. These files can be used to reconstruct all probe penetrations. 

## `fano`

Within the 'cv1.0' (cutoff used to include cells being a coefficient of variation of 1.0 across ten chunks of the recording session; see Methods) folder and protocol-specific folders, the `{protocol}_results.mat` file contains the output of `Variance_toolbox`. It is a cell array called Results with shape `(n_sessions, n_trace_types)`. Each cell is a struct containing the `Variance_toolbox` output for an individual session and trial type (odor). This struct contains field-value pairs, where each value has length `n_neurons` recorded during that section. Documentation is available [here](https://www.dropbox.com/scl/fo/uoqaoli4mrd7w6j95y8ko/AMq2G-M6Y5OUBNKX6zSRf4E?rlkey=hlsaw999h4p9ijhsycmc1aizt&e=1&dl=0).

In addition to `{protocol}_results.mat`, there may also be subfolders (e.g. `AL39/20211001`) containing .png files with the results of the `Variance_toolbox` analysis on particular mice and sessions. Each unique .png is for a different trial type (odor). Those with 'scatter' in the name show the Fano factor analysis for a particular time slice, as in Churchland et al., 2010, Fig. 4b. Those with 'tt' in the name show the results of this analysis for multiple time slices spanning the full trial, as in Churchland et al., 2010, Fig. 4c. They are included because the raw data used to generate them is too large to be uploaded, and so while they can't be regenerated from scratch, they are representative of all sessions. 

## `glm`

This folder contains three types of files.

Those that end with 'neuron_info.sav' (e.g. 'SameRewDist_ephys_combined_spks_glm_neuron_info.sav') contain a dictionary that is meant to be recast as a pandas DataFrame, and is used by `neural_analysis/glm_analysis.ipynb`. For 'ephys' tables, this DataFrame will have 20 columns, while for 'imaging' tables, it will have 11 columns. The number of rows is equal to the number of neurons recorded (each row describes a different neuron). The columns are as follows:

### `ephys`

- `names`: mouse name
- `file_dates`: the date on which this neuron was recorded
- `cluster_id`: the cluster_id given by Phy of included cells. Included cells were manually labeled "good" in Phy and then also subject to minimum firing rate and coefficient of variation quality controls; see Methods.
- `neuron_idx_good`: the index (row number, 0-indexed of cluster_info.tsv file) of the cells that Phy labels "good" and that meet the inclusion criteria. Rarely used.
- `neuron_idx_inc`: of just the cells marked "good" in Phy, these are the indices (from `np.flatnonzero()`) of those cells that subsequently meet the inclusion criteria. Not the same as `neuron_idx_good` because rows will be skipped that are not labeled "good" in Phy.
- `fig_paths`: path to raw data (not operable)
- `depths`: DV coordinate (relative to bregma, in mm)
- `aps`: AP coordinate (relative to bregma, in mm)
- `mls` ML coordinate (relative to bregma, in mm)
- `regions`: Allen CCF region
- `region_ids`: Allen CCF region ID
- `kim_regions`: region according to the Kim mouse brain atlas (Chon et al., 2019), which introduces a finer parcellation of striatal subregions.
- `kim_region_ids`: region ID according to the Kim mouse brain atlas
- `kim_generals`: region according to the Kim mouse brain atlas at hierarchy level 6 (coarser)
- `means`: mean of firing rate trace
- `stds`: standard deviation of firing rate trace
- `cvs`: coefficient of variation of firing rate trace
- `kurtosis`: kurtosis of firing rate trace
- `cell_types`: putative cell type ID ('MSN', 'FSI', 'axonal', 'TAN')
- `genotype`: genotype of animal from which it came

### `imaging`

- `names`: mouse name
- `file_dates`: the date on which this neuron was recorded
- `neuron_idx_inc`: of just the cells marked "good" in Phy, these are the indices (from `np.flatnonzero()`) of those cells that subsequently meet the inclusion criteria. Not the same as `neuron_idx_good` because rows will be skipped that are not labeled "good" in Phy.
- `fig_paths`: path to raw data (not operable)
- `genotype`: genotype of animal from which it came (D1-Cre or A2a-Cre)
- `zF_means`: mean of z-scored fluorescence trace (0)
- `zF_stds`: standard deviation of z-scored fluorescence trace (1)
- `dFF_means`: mean of delta F/F trace (0)
- `dFF_stds`: standard deviation of delta F/F trace
- `spks_means`: mean of deconvolved activity trace
- `spks_stds`: standard deviation of deconvolved activity trace

In addition to the 'neuron_info.sav' files, there are two more file types. Those ending with `glm_model_specs` contain the information used to fit each GLM. Note that GLMs fit all neurons in an entire session simultaneously, but that multiple GLMs might be fit on the same session with different options, e.g. the full model vs. dropping out motor regressors. They each contain a pandas DataFrame of shape `(n_models, 28)` with the following 28 columns, many of which are input directly to [GLM_Tensorflow_2](https://github.com/sytseng/GLM_Tensorflow_2/tree/main):

- `mid`: mouse ID
- `name`: mouse name
- `sid`: session ID
- `rid`: run ID
- `figure_path`: path to raw data (not operable)
- `exp_date`: experiment date
- `i_model`: counter for how many models were run on the same session/dt
- `protocol`: task (e.g. SameRewDist)
- `fit_date`: date the GLM was fit
- `n_trial`: number of trials in that session
- `n_cells`: number of cells included from that session
- `n_folds`: number of folds for cross-validation (5)
- `auto_split`: perform CV split automatically or not, bool, default = True
- `split_by_group`: perform CV split according to a third-party provided group when auto_split = True, default = True
- `activation`: activation function ('exp' for Poisson GLM)
- `loss_type`: pointwise deviance loss (default = 'poisson')
- `regularization`: group_lasso or elastic_net
- `lambda_series`: which lambda values were tried during cross-validation, in order
- `optimizer`: which optimizer was used for fitting (always Adam)
- `learning_rate`: learning rate for Adam optimizer
- `fitting_time`: time it took to fit model
- `dropped_out_vars`: which variables were dropped out on this run (e.g. 'none', 'motor', etc.)
- `regressor_labels`: names of each regressor, in order
- `se_frac`: choose the highest lambda value that is within `se_frac` * standard_error of the lambda that minimizes CV deviance explained, to apply to the test set.
- `l1_ratio`: when using elastic_net, how much l1 penalty to use (between 0 and 1)
- `dt`: width used for binning spikes (e.g. 20 ms)
- `seed`: random seed used for assigning train/test and CV splits
- `modality`: ephys or imaging

The last file type ends in 'glm_fit_table.sav', and it stores the fitting results. It has shape `(n_neurons, 56)`, many columns of which are identical to `glm_model_specs`. Unique columns are as follows:

- `i_cell`: cell ID from that session
- `lambda`: the value of lambda that was selected
- `lambda_ind`: the index of the lambda that was selected (such that `lambda = lambda_series[lambda_ind]`)
- `null_dev`: null deviance
- `full_coefs`: coefficients from the full model (for the order of the regressors, find the corresponding entry in glm_model_specs) 
- `full_dev`: deviance of the full model
- `full_dev_expl`: deviance explained of the full model
- `full_dev_abl_nuissance`: deviance of the full model after zeroing the nuissance regressors
- `full_dev_expl_abl_nuissance`: deviance explained of the full model after zeroing the nuissance regressors
- `expectiles_coefs`: coefficients from the model in which expectiles (and redundant odor responses) have been dropped out (drop_expectiles)
- `expectiles_dev`: deviance of the drop_expectiles model
- `expectiles_dev_expl`: deviance explained of the drop_expectiles model
- `expectiles_dev_abl_nuissance`: deviance of the drop_expectiles model after zeroing the nuissance regressors
- `expectiles_dev_expl_abl_nuissance`: deviance explained of the drop_expectiles model after zeroing the nuissance regressors
- `expectiles_frac_expl_dev`: fraction of explained deviance of the drop_expectiles model. We evaluate the model deviance of the original model ("full model") and that of the dropped-out model. We then compute the difference between the model deviance, and normalize it by the explained deviance of the full model to compute the "fraction explained deviance". (Don't confuse this term with "fraction deviance explained" when we report model performance; "fraction deviance explained" means how much null deviance is explained by the full model, whereas "fraction explained deviance" means how much of the explained deviance of the full model is contributed by a the variable we remove in the ablated model.)
- `expectiles_frac_null_dev`: fraction of null deviance of the drop_expectiles model. This is identical to `expectiles_frac_expl_dev` above, except that instead of normalizing by the explained deviance of the full model, we normalize by the null deviance.

The last seven columns (those beginning with `expectiles`) are then each repeated four times, for the models in which `licking`, `motor`, `history`, and `reward` regressors are dropped out instead.

## `imaging`

This folder contains three subfolders: 'AL60', 'AL65', and 'tmp' (the latter of which is empty). The subject-specfic folders contain the outputs of suite2p on three example sessions, which are used for plotting the individual example neurons in Fig. 4. Refer to the suite2p documentation for details.