import six
import numpy as np
from scipy.io import loadmat

from geo_tool import Point_Cloud

segs_extension = '_segs.txt'
feat_extension = '_feat.txt'
points_extension = '_pts.txt'
points_w_normals_extension = '_pts_and_normals.txt'


################################################
# gradient files: model_id\n
#                 bneck_layer_size\n
#                 point_0 gx gy gz ... gx gy gz\n
#                 ...
#                 point_N gx gy gz ... gx gy gz\n
# extension = _grad.txt
################################################


shape_net_core_synth_id_to_category = {
    '02691156': 'airplane', '02773838': 'bag',
    '02801938': 'basket', '02808440': 'bathtub', '02818832': 'bed',
    '02828884': 'bench', '02834778': 'bicycle', '02843684': 'birdhouse',
    '02871439': 'bookshelf', '02876657': 'bottle', '02880940': 'bowl',
    '02924116': 'bus', '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair', '03046257': 'clock',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03207941': 'dishwasher', '03211117': 'display', '03261776': 'earphone',
    '03325088': 'faucet', '03337140': 'file', '03467517': 'guitar',
    '03513137': 'helmet', '03593526': 'jar', '03624134': 'knife',
    '03636649': 'lamp', '03642806': 'laptop', '03691459': 'speaker',
    '03710193': 'mailbox', '03759954': 'microphone', '03761084': 'microwave',
    '03790512': 'motorcycle', '03797390': 'mug', '03928116': 'piano',
    '03938244': 'pillow', '03948459': 'pistol', '03991062': 'pot',
    '04004475': 'printer', '04074963': 'remote_control', '04090263': 'rifle',
    '04099429': 'rocket', '04225987': 'skateboard', '04256520': 'sofa',
    '04330267': 'stove', '04530566': 'vessel', '04554684': 'washer'
}


def shape_net_category_to_synth_id():
    d = shape_net_core_synth_id_to_category
    inv_map = {v: k for k, v in six.iteritems(d)}
    return inv_map


def match_incomplete_to_complete_data(complete_model_names, incomplete_model_names):
    # Match incomplete to complete model_names.
    d = {name: i for i, name in enumerate(complete_model_names)}
    mapping = []    # The i-th incomplete pcloud will correspond to the mapping[i] in the order of the complete_model_names.
    for name in incomplete_model_names:
        mapping.append(d[name])
    mapping = np.array(mapping)
    return np.array(mapping)


def syn_id_to_class_id_dict():
    d1 = shape_net_core_synth_id_to_category
    return {key: i for i, key in enumerate(d1)}  # Map syn-ids to integers


def map_syn_ids_to_class_ids(syn_ids):
    d = syn_id_to_class_id_dict()
    return np.array([d[s] for s in syn_ids])


def find_models_in_category(class_ids, category_id):
    syn_id = shape_net_category_to_synth_id()[category_id]
    class_id = syn_id_to_class_id_dict()[syn_id]
    return np.where(class_ids == class_id)[0]


def model_ids_to_syn_ids(model_names_alinged, labels_alinged, query_names):
    d = dict()
    for m, l in zip(model_names_alinged, labels_alinged):
        d[m] = l
    return [d[name] for name in query_names]


def model_ids_to_class_ids(all_model_names, all_syn_ids, query_names):
    ''' all_model_names,  all_syn_ids: 2 np arrays that are aligned and contain names and syn_ids respectively.
    '''
    return map_syn_ids_to_class_ids(model_ids_to_syn_ids(all_model_names, all_syn_ids, query_names))


def occupancy_grid_as_point_cloud(matlab_file):
    '''Visualize occupancy grids saved as matlab files.
    '''
    data = loadmat(matlab_file)
    data = data['grid']     # Matlab variable name used.
    x, y, z = np.where(data)
    points = np.vstack((x, y, z)).T
    return Point_Cloud(points=points)
