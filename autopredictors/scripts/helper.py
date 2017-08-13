import six
import numpy as np
from scipy.io import loadmat

from geo_tool import Point_Cloud

segs_extension = '_segs.txt'
feat_extension = '_feat.txt'
points_extension = '_pts.txt'
points_w_normals_extension = '_pts_and_normals.txt'


# def match_incomplete_to_complete_data(complete_model_names, incomplete_model_names):
#     # Match incomplete to complete model_names.
#     d = {name: i for i, name in enumerate(complete_model_names)}
#     mapping = []    # The i-th incomplete pcloud will correspond to the mapping[i] in the order of the complete_model_names.
#     for name in incomplete_model_names:
#         mapping.append(d[name])
#     mapping = np.array(mapping)
#     return np.array(mapping)

# def syn_id_to_class_id_dict():
#     d1 = shape_net_core_synth_id_to_category
#     return {key: i for i, key in enumerate(d1)}  # Map syn-ids to integers
# 
# 
# def map_syn_ids_to_class_ids(syn_ids):
#     d = syn_id_to_class_id_dict()
#     return np.array([d[s] for s in syn_ids])
# 
# 
# def find_models_in_category(class_ids, category_id):
#     syn_id = shape_net_category_to_synth_id()[category_id]
#     class_id = syn_id_to_class_id_dict()[syn_id]
#     return np.where(class_ids == class_id)[0]
# 
# 
# def model_ids_to_syn_ids(model_names_alinged, labels_alinged, query_names):
#     d = dict()
#     for m, l in zip(model_names_alinged, labels_alinged):
#         d[m] = l
#     return [d[name] for name in query_names]
# 
# 
# def model_ids_to_class_ids(all_model_names, all_syn_ids, query_names):
#     ''' all_model_names,  all_syn_ids: 2 np arrays that are aligned and contain names and syn_ids respectively.
#     '''
#     return map_syn_ids_to_class_ids(model_ids_to_syn_ids(all_model_names, all_syn_ids, query_names))


def occupancy_grid_as_point_cloud(matlab_file, var_name='grid'):
    '''Visualize an occupancy grid saved in a matlab files, with the
    variable holding it named as 'var_name'.
    '''
    data = loadmat(matlab_file)
    data = data[var_name]     # Matlab variable name used.
    x, y, z = np.where(data)
    points = np.vstack((x, y, z)).T
    return Point_Cloud(points=points)
