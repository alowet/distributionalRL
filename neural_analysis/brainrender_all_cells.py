import numpy as np

from brainrender import Scene
from brainrender.actors import Points
from brainrender.video import VideoMaker

import seaborn as sns
import joblib
import pickle
import os
import glob
import sys
sys.path.append('../utils')
from db import get_db_info
from analysisUtils import assign_str_regions_from_kim

# protocols = ['DistributionalRL_6Odours', 'Bernoulli', 'SameRewDist']
protocols = ['SameRewDist']
manipulation = 'combined'

paths = get_db_info()
# data_dir = os.path.join(paths['neural_fig_roots'][1], 'pooled', 'preloaded')
data_dir = '/mnt/hdd1/dist-rl/data/to_qiao/20240828/'
print(data_dir)
# atlas = 'kim_mouse_50um'
atlas = 'allen_mouse_25um'

# Create a brainrender scene
scene = Scene(screenshots_folder='./brainrender_plots/screenshots', inset=False, atlas_name=atlas)

# Add brain regions
cp = scene.add_brain_region('CP', alpha=0.2, color='dr')
acb = scene.add_brain_region('ACB', alpha=0.2, color='dr')

# regions = assign_str_regions_from_kim(None)
# for i_reg, reg_list in enumerate(regions.values()):
#     if type(reg_list) == str:
#         reg_list = [reg_list]
#     print(reg_list)
#     scene.add_brain_region(*reg_list, alpha=0.7, color=sns.color_palette()[i_reg], hemisphere='both')

for protocol in protocols:

    # pfiles = glob.glob(os.path.join(data_dir, protocol + '_ephys_{}_striatum_spks_*.sav'.format(manipulation)))
    pfiles = glob.glob(os.path.join(data_dir, protocol + '_{}_spks_data.p'.format(manipulation)))
    # latest_file = max(pfiles, key=os.path.getctime)
    # with open(latest_file, 'rb') as f:
    assert len(pfiles) == 1  # asserts that I'm using the correct saved file. Will need to change if I have multiple
    with open(pfiles[0], 'rb') as f:
        proc_data = pickle.load(f)

    # load coordinates
    print(proc_data.keys())
    data = proc_data['neuron_info']
    bregma_xyz = np.stack([data[x] for x in ['aps', 'depths', 'mls']], axis=1)
    ncells = bregma_xyz.shape[0]

    # Choose color of points
    colors = np.full((ncells, 3), dtype='object', fill_value=[0.8, 0.8, 0.8])

    #  Add points to scene, colored by subregion
    # neuron_info, reg_labels = assign_str_regions_from_kim({i: data[i] for i in data if i != 'sub_inds'})
    # color_cycle = sns.color_palette()
    # for i_reg, reg in enumerate(reg_labels):
    #     colors[neuron_info['str_regions'] == reg] = color_cycle[i_reg]

    # put random half in each hemisphere
    rng = np.random.default_rng()
    bregma_xyz[:, 2] = bregma_xyz[:, 2] * rng.choice([-1, 1], size=ncells)

    # bregma in CCF coordinates from https://int-brain-lab.github.io/iblenv/_modules/ibllib/atlas/atlas.html
    # required order is AP, DV, ML (https://docs.brainrender.info/usage/using-your-data/registering-data)
    coordinates = ([5400, 332, 5739] - bregma_xyz * 1000)

    # gmm_subset = np.flatnonzero(np.load(os.path.join(data_dir, 'value_gmm_' + protocol + '.npy')))
    # value_tercile = np.flatnonzero(np.load(os.path.join(data_dir, 'upper_Value_tercile_' + protocol + '.npy')))
    # licking_tercile = np.flatnonzero(np.load(os.path.join(data_dir, 'upper_Licking_tercile_' + protocol + '.npy')))

    # for keystr in ['all_value_correl_']:  # ['value_dropout_']:  # ['mnemonic_subspace_']:  # ['value_gmm_', 'upper_Value_tercile_', 'upper_Licking_tercile_']:
    #     # subset = np.flatnonzero(np.load(os.path.join(data_dir, keystr + protocol + '_ephys.npy')))
    #     subset = data['sub_inds']['all_value_correl']


    # Add points to scene, colored by value coding
    # colors = np.full((ncells, 3), dtype='object', fill_value=[0.6, 0.6, 0.6])  #fill_value='steelblue')
    # colors[data['value_subset']] = 'dr'
    # colors[subset] = [255, 0, 255]

    scene.add(Points(coordinates + np.random.normal(0, [20, 0, 20], coordinates.shape), name="cells",
                     radius=40, alpha=0.2, res=15, colors=colors))

    # print(bregma_xyz.shape)
    # print(np.amax(bregma_xyz[:, 0]))

# render
scene.render(interactive=False, zoom=1)
# scene.screenshot(name='all_cells_{}'.format(atlas))
scene.screenshot(name='srd_cells_{}'.format(atlas))

# Create an instance of video maker
# vm = VideoMaker(scene, "./brainrender_plots/videos", 'all_cells_{}'.format(atlas))
vm = VideoMaker(scene, "./brainrender_plots/videos", 'srd_cells_{}'.format(atlas))

# this just rotates the scene
vm.make_video(azimuth=2, duration=10)