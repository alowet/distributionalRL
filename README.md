# distributionalRL

This repo contains the code for generating figures in Lowet et al. (in press). In order to run, it is also necessary to download the data from Zenodo, which should be placed in a directory alongside the code folder as follows (this repo is equivalent to the `code` folder):

```
├── parent_dir
│   ├── code
│   ├── ├── neural_analysis
│   ├── ├── behavior_analysis
│   ├── ├── ann_decoding
│   ├── ├── ...
│   ├── data
```
## Organization

The code is formatted as Jupyter notebooks. There are ten such notebooks, each located within the relevant subfolder.

1. `neural_analysis/recording_figs.ipynb` plots neural recording data (mostly Figs. 2, 3, 4 and ED Figs. 2-4, 6-8, and 10d-e.
2. `behavior_analysis/compare_optostim.ipynb` plots optogenetic stimulation data (Fig. 5 and ED Fig. 7a-m, 11).
3. `behavior_analysis/licking_all_sessions.ipynb` plots licking data (Fig. 1c, ED Fig. 1f-g, 8b, 10c).
4. `behavior_analysis/behavioral_decoding.ipynb` plots (Fig. 1d, ED Fig. 1h, 9b-c).
5. `neural_analysis/plot_smoothed_data.ipynb` plots Fig. 1f-g., ED Fig. 2b, 10a-b
6. `neural_analysis/plot_sample_data.ipynb` plots Fig. 1h, 4d,g.
7. `neural_analysis/glm_analysis.ipynb` plots data from ED Fig. 5 and 9d-f. TODO
8. `neural_analysis/compare_fano.ipynb` plots Fano factor analysis (ED Fig. 6). TODO
9. `ann_decoding/ann_decoding.ipynb` plots data from ANN-based decoding (ED Fig. 4f-l).
10. `behavioral_analysis/plot_facemap_components.ipynb` plots ED Fig. 1e.

## Description of `data`

At the topmost level, `data` contains the SQLITE database with information about all mice, behavior sessions, ephys/imaging sessions, etc., saved as separate tables. This will be queried frequently to pull up the relevant sessions for analysis. Beyond this, `data` contains several subfolders:

`neural-plots` contains files with ephys and imaging data for the various protocols. The main task is called SameRewDist, and the files are called e.g. `SameRewDist_ephys_combined_striatum_spks.sav`. To open these files, use Python's `joblib` package. Each of the files contains a `dict` with the following keys:

- 'all_spk_cnts': an array of spike counts binned every 250 ms, shape (n_trial_types, total_cells, max_n_trials_per_type, n_psth_bins). max_n_trials_per_type was set to 90; unused trials are np.nan. n_trial_types = n_trace_types + 1, as it includes  Unexpected Reward trials.
- 'cue_resps': an array of spike counts binned every 1 s, shape (n_trace_types, total_cells, max_n_trials_per_type, n_prerew_periods). n_prerew_periods = n_periods - 1 = 4, as it excludes Outcome.
- 'X_means': an array of trial-average firing rates, shape (n_trial_types, total_cells, n_periods)
- 'cell_stds': standard deviation of each cell's firing rate, shape (n_trial_types, total_cells, n_periods)
- 'neuron_info': dictionary (for pickling purposes) which can and should be restructured as a DataFrame using pd.DataFrame(neuron_info) with shape (total_cells, 23). Contains information like mouse, coordinates, mean, std, `class_name`, and striatal subregion (str_regions) for each neuron in `all_spk_cnts`, `cue_resps`, etc. in order.
'late_trace_ind': equal to 3, which period index (e.g. of `X_means` or `cue_resps`) corresponds to the Late Trace period
- 'n_trace_types': protocol-specific, usually 6 (for 6 odors used), but sometimes 5 (e.g. Bernoulli task), for the number of odors with trace periods
- 'psth_bin_width': 250 ms, bin width for PSTH
- 'psth_bin_centers': centers of PSTH bins
- 'corrs': correlations with e.g. mean, reward, rpe, etc, computed at psth_bin_width with psth_bin_centers
- 'corrs_seconds': correlations with e.g. mean, reward, rpe, etc, computed during each period (seconds)
- 'n_psth_bins': len(psth_bin_centers)
- 'n_periods': equal to 5, the number of periods used, each 1 s long (Baseline, Odor, Early Trace, Late Trace, Outcome)
- 'total_cells': number of cells, len(neuron_info)
