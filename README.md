# distributionalRL

This repo contains the code for generating figures in Lowet et al. (in press). In order to run, it is also necessary to download the data from Zenodo, which should be placed in a directory alongside the code folder as follows:
- parent_dir
- -code
- -data

## Organization

The code is formatted as Jupyter notebooks. There are __ such notebooks, each located within the relevant subfolder.

1. `neural_analysis/recording_figs.ipynb` plots neural recording data (mostly Figs. 2, 3, 4 and ED Figs. 2-4 and 6-8.
2. `behavior_analysis/compare_optostim.ipynb` plot optogenetic stimulation data (Fig. 5 and ED Fig. 7a-m, 11)
3. `behavior_analysis/licking_all_sessions.ipynb` plots licking data (Fig. 1c, ED Fig. 1f-g, 8b, 10c).
4. `behavior_analysis/behavioral_decoding.ipynb` plots (Fig. 1d, ED Fig. 1h, 9b-c).
5. `ann_decoding/ann_decoding.ipynb` plots data from ANN-based decoding (ED Fig. 4f-l).
6. `neural_analysis/glm_analysis.ipynb` plots data from ED Fig. 5 and 9d-f.
7. `behavioral_analysis/plot_facemap_components.ipynb` plots ED Fig. 1e.
8. `neural_analysis/smooth_data.ipynb` plots Fig. 1f-g.
9. `neural_analysis/sample_data.ipynb` plots Fig. 1h, 4d,g.