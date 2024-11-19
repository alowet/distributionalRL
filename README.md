# distributionalRL

This repo contains the code for generating figures in Lowet et al. (in press). In order to run, it is also necessary to download the data from [Dryad](https://doi.org/10.5061/dryad.80gb5mm0m), which should be placed in a directory alongside the code folder as follows (this repo is equivalent to the `code` folder):

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

The necessary conda environment for running the code is available in `envs/environment.yml`. For `glm_analysis`, the `tf.yml` file environment should be used instead. To do so, ensure [conda is installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and then execute the command `conda env create -f environment.yml`.

In order to match the plotting style used in the paper, it is necessary to copy the `paper_export.mplstyle` file (found in the `utils` folder) to your environment-specific `matplotlib` stylelib folder. On my computer, this is found in `~/anaconda3/envs/neural/lib/python3.8/site-packages/matplotlib/mpl-data/stylelib/`, although yours may differ depending on your conda installation.

## Organization

The code is formatted as Jupyter notebooks. There are ten such notebooks, each located within the relevant subfolder.

1. `neural_analysis/recording_figs.ipynb` plots neural recording data (mostly Figs. 2, 3, 4 and ED Figs. 2-4, 6-8, and 10d-e.
2. `behavior_analysis/compare_optostim.ipynb` plots optogenetic stimulation data (Fig. 5 and ED Fig. 7a-m, 11).
3. `behavior_analysis/licking_all_sessions.ipynb` plots licking data (Fig. 1c, ED Fig. 1f-g, 8b, 10c).
4. `behavior_analysis/behavioral_decoding.ipynb` plots (Fig. 1d, ED Fig. 1h, 9b-c).
5. `neural_analysis/plot_smoothed_data.ipynb` plots Fig. 1f-g., ED Fig. 2b, 10a-b
6. `neural_analysis/plot_sample_data.ipynb` plots Fig. 1h, 4d,g.
7. `neural_analysis/glm_analysis.ipynb` plots data from ED Fig. 5 and 9d-f.
8. `neural_analysis/compare_fano.ipynb` plots Fano factor analysis (ED Fig. 6).
9. `ann_decoding/ann_decoding.ipynb` plots data from ANN-based decoding (ED Fig. 4f-l).
10. `behavioral_analysis/plot_facemap_components.ipynb` plots ED Fig. 1e.

They should all run after following the instructions above. If they don't, post an issue and I'll investigate!