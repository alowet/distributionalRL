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

# protocols = ['DistributionalRL_6Odours', 'Bernoulli', 'SameRewDist']
protocols = ['SameRewDist']
manipulation = 'combined'

paths = get_db_info()
data_dir = '../data/'
print(data_dir)
atlas = 'allen_mouse_25um'

# Create a brainrender scene
scene = Scene(screenshots_folder='./brainrender_plots/screenshots', inset=False, atlas_name=atlas)

# Add brain regions
cp = scene.add_brain_region('CP', alpha=0.2, color='dr')
acb = scene.add_brain_region('ACB', alpha=0.2, color='dr')

for protocol in protocols:

    pfiles = glob.glob(os.path.join(data_dir, protocol + '_ephys_{}_striatum_spks_*.sav'.format(manipulation)))
    # pfiles = glob.glob(os.path.join(data_dir, protocol + '_{}_spks_data.p'.format(manipulation)))
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

    #  Add points to scene, putting random half in each hemisphere
    rng = np.random.default_rng()
    bregma_xyz[:, 2] = bregma_xyz[:, 2] * rng.choice([-1, 1], size=ncells)

    # bregma in CCF coordinates from https://int-brain-lab.github.io/iblenv/_modules/ibllib/atlas/atlas.html
    # required order is AP, DV, ML (https://docs.brainrender.info/usage/using-your-data/registering-data)
    coordinates = ([5400, 332, 5739] - bregma_xyz * 1000)

    scene.add(Points(coordinates + np.random.normal(0, [20, 0, 20], coordinates.shape), name="cells",
                     radius=40, alpha=0.2, res=15, colors=colors))

# render
scene.render(interactive=False, zoom=1)
# scene.screenshot(name='all_cells_{}'.format(atlas))
scene.screenshot(name='srd_cells_{}'.format(atlas))

# Create an instance of video maker
vm = VideoMaker(scene, "./brainrender_plots/videos", 'srd_cells_{}'.format(atlas))

# this just rotates the scene
vm.make_video(azimuth=2, duration=10)