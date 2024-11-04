from brainrender import Scene
from brainrender.video import VideoMaker
from brainrender.actor import Actor

from vedo import shapes
from loguru import logger

import numpy as np
import pandas as pd
import json
import os
import sys
sys.path.append('../utils')
from db import get_db_info, execute_sql, update_db

# Very similar to cylinder, but expects pos to be two points, top and bottom
class Probe(Actor):
    def __init__(self, top, bottom, root, color="black", alpha=1, radius=10, res=24):
        """
        Cylinder class creates a cylinder mesh between a given
        point and the brain's surface.
        :param top: list, np.array in ap, dv, ml coordinates.
        :param bottom: list, np.array in ap, dv, ml coordinates.
        :param root: brain root Actor or mesh object
        :param color: str, color
        :param alpha: float
        :param radius: float
        """

        logger.debug(f"Creating Cylinder actor at: {top, bottom}")

        # Create mesh and Actor
        mesh = shapes.Cylinder(pos=[top, bottom], c=color, r=radius, alpha=alpha, res=res)
        Actor.__init__(self, mesh, name="Probe", br_class="Probe")


# protocols = ['DistributionalRL_6Odours', 'Bernoulli', 'SameRewDist']
protocols = ['SameRewDist']
manipulation = 'combined'
perm_order = [0, 2, 1]
# ccf_perm = [1, 2, 0]

paths = get_db_info()
data_dir = os.path.join(paths['neural_fig_roots'][1], 'pooled', 'preloaded')
# atlas = 'kim_mouse_50um'
atlas = 'allen_mouse_25um'

# Create a brainrender scene
scene = Scene(screenshots_folder='./brainrender_plots/screenshots', inset=False, atlas_name=atlas)

# Add brain regions
cp = scene.add_brain_region('CP', alpha=0.1, color='blue')
acb = scene.add_brain_region('ACB', alpha=0.1, color='green')

bregma = np.array([5400, 332, 5739])  # y, z, x (ap, dv, ml) as demanded by brainrender.actors.cylinder

for protocol in protocols:

    sql = 'SELECT e.name, file_date_id, ccf_xyz FROM ephys AS e LEFT JOIN session AS s ON e.name=s.name AND e.file_date=s.exp_date WHERE ' + \
        '(e.name IN(SELECT name FROM mouse WHERE surgery1="headplate") OR e.probe1_ML < 0) AND ' + \
        's.significance=1 AND protocol="{}" AND registered=1'.format(protocol)
    ccf_xyzs = execute_sql(sql, paths['db'])
    df = pd.DataFrame(ccf_xyzs, columns=ccf_xyzs[0].keys())

    for index, row in df.iterrows():

        # if row['ccf_xyz'] is None:
        #     # bregma in CCF coordinates from https://int-brain-lab.github.io/iblenv/_modules/ibllib/atlas/atlas.html
        #     # required order is AP, DV, ML (https://docs.brainrender.info/usage/using-your-data/registering-data)
        #     # right now, point is in ML, AP, DV
        #     xyz_picks_file = os.path.join('/mnt/nas/ephys', row['name'], row['file_date_id'], 'alf', 'xyz_picks.json')
        #     try:
        #         with open(xyz_picks_file, 'rb') as f:
        #             bregma_xyz = json.load(f)
        #         entries = [[list([5400, 332, 5739] - np.array(point)[perm_order]) for point in bregma_xyz]]
        #         # update_sql = 'UPDATE ephys SET ccf_xyz = "{}" WHERE name = "{}" AND file_date_id = "{}"'.format(
        #         #     json.dumps(entries[0]), row['name'], row['file_date_id'])
        #     except FileNotFoundError:
        #         # continue
        #         ccf_xyz = {}
        #         entries = []
        #         for ifile in range(1, 5):
        #             xyz_picks_file = os.path.join('/mnt/nas/ephys', row['name'], row['file_date_id'], 'alf', 'xyz_picks_shank{}.json'.format(ifile))
        #             with open(xyz_picks_file, 'rb') as f:
        #                 bregma_xyz = json.load(f)
        #             ccf_xyz['shank{}'.format(ifile)] = []
        #             for point in bregma_xyz:
        #                 ccf_xyz['shank{}'.format(ifile)].append(list([5400, 332, 5739] - np.array(point)[perm_order]))
        #             entries.append(ccf_xyz['shank{}'.format(ifile)])
        #         # update_sql = "UPDATE ephys SET ccf_xyz = '{}' WHERE name = '{}' AND file_date_id = '{}'".format(
        #         #     json.dumps(ccf_xyz), row['name'], row['file_date_id'])
        #     # UPDATES DON'T WORK AS WRITTEN BECAUSE ORDER IS WRONG FOR DB
        #     # update_db(update_sql, paths['db'])
        #
        # else:
        #     ccf_xyz = json.loads(row['ccf_xyz'])
        #     entries = []
        #     if type(ccf_xyz) == dict:
        #         for v in ccf_xyz.values():
        #             entries.append(list([np.array(x)[perm_order] for x in v]))
        #     else:
        #         entries.append(list([np.array(x)[perm_order] for x in ccf_xyz]))

        entries = []
        try:
            channel_locs_file = os.path.join('/mnt/nas2/ephys', row['name'], row['file_date_id'], 'alf',
                                             'channel_locations.json')
            with open(channel_locs_file, 'rb') as f:
                locs = json.load(f)
            entries = [[np.array([point['y'], point['z'], -point['x']]) for point in [locs['channel_0'], locs['channel_382']]]]
        except FileNotFoundError:
            for ifile in range(1, 5):
                channel_locs_file = os.path.join('/mnt/nas2/ephys', row['name'], row['file_date_id'], 'alf', 'channel_locations_shank{}.json'.format(ifile))
                with open(channel_locs_file, 'rb') as f:
                    locs = json.load(f)
                entries.append([np.array([point['y'], point['z'], -point['x']]) for point in [locs['channel_0'], locs['channel_94']]])

        entries = bregma - np.array(entries)
        for entry in entries:
            actor = Probe(*entry, scene.root, alpha=.2, radius=16, color=[.2]*3, res=50)
            scene.add(actor)


# render
scene.render(interactive=False, zoom=1)
scene.screenshot(name='srd_probes_{}'.format(atlas))

# # Create an instance of video maker
# vm = VideoMaker(scene, "./brainrender_plots/videos", 'srd_probes_{}'.format(atlas))
# #
# # # this just rotates the scene
# vm.make_video(azimuth=2, duration=8)

