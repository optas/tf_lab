import sys
import glob
import os.path as osp
import numpy as np
from scipy.io import loadmat

from geo_tool import Point_Cloud

segs_extension = '_segs.txt'
feat_extension = '_feat.txt' 
points_extension = '_pts.txt'


################################################
# gradient files: model_id\n
#                 bneck_layer_size\n
#                 point_0 gx gy gz ... gx gy gz\n
#                 ...
#                 point_N gx gy gz ... gx gy gz\n
# extension = _grad.txt
################################################


shape_net_core_synth_id_to_category = {
    '02691156': 'airplane',   '02747177': 'can',            '02773838': 'bag',
    '02801938': 'basket',     '02808440': 'bathtub',        '02818832': 'bed',
    '02828884': 'bench',      '02834778': 'bicycle',        '02843684': 'birdhouse',
    '02871439': 'bookshelf',  '02876657': 'bottle',         '02880940': 'bowl',
    '02924116': 'bus',        '02933112': 'cabinet',        '02942699': 'camera',
    '02946921': 'tin_can',    '02954340': 'cap',            '02958343': 'car',
    '03001627': 'chair',      '03046257': 'clock',          '03085013': 'keyboard',
    '03207941': 'dishwasher', '03211117': 'display',        '03261776': 'earphone',
    '03325088': 'faucet',     '03337140': 'file',           '03467517': 'guitar',
    '03513137': 'helmet',     '03593526': 'jar',            '03624134': 'knife',
    '03636649': 'lamp',       '03642806': 'laptop',         '03691459': 'speaker',
    '03710193': 'mailbox',    '03759954': 'microphone',     '03761084': 'microwave',
    '03790512': 'motorcycle', '03797390': 'mug',            '03928116': 'piano',
    '03938244': 'pillow',     '03948459': 'pistol',         '03991062': 'pot',
    '04004475': 'printer',    '04074963': 'remote_control', '04090263': 'rifle',
    '04099429': 'rocket',     '04225987': 'skateboard',     '04256520': 'sofa',
    '04330267': 'stove',      '04379243': 'table'
}


def shape_net_category_to_synth_id():
    d = shape_net_core_synth_id_to_category
    inv_map = {v: k for k, v in d.iteritems()}
    return inv_map


def occupancy_grid_as_point_cloud(matlab_file):
    '''Visualize occupancy grids saved as matlab files.
    '''
    data = loadmat(matlab_file)
    data = data['grid']     # Matlab variable name used.
    x, y, z = np.where(data)
    points = np.vstack((x, y, z)).T
    return Point_Cloud(points=points)
