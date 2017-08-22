'''
Created on February 27, 2017

@author: optas

 Data used by Dai et al. in
'''

import struct
import numpy as np
import os.path as osp
from collections import defaultdict

from geo_tool import Point_Cloud
from general_tools.simpletons import indices_in_iterable

from ... point_clouds.in_out import load_filenames_of_input_data, load_crude_point_clouds, PointCloudDataSet, train_validate_test_split
from . helper import points_extension

all_classes = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']

# Plotting Parameters for ICCV submission.
in_u_sphere_plotting = {'chair': True, 'airplane': False, 'cabinet': False, 'car': False, 'lamp': True, 'sofa': True, 'table': True, 'vessel': False}

azimuth_angles = {'chair': -50, 'airplane': 0, 'cabinet': -40, 'car': -60, 'lamp': 0, 'sofa': -60, 'table': 60, 'vessel': -60}

plotting_color = {'chair': 'g', 'airplane': 'b', 'cabinet': 'orange', 'car': 'r', 'lamp': 'yellow', 'sofa': 'magenta', 'table': [0.7, .45, 0], 'vessel': [0, 0.6, 1]}


def plotting_default_params(category):
    kwdict = {}
    kwdict['show_axis'] = False
    kwdict['in_u_sphere'] = in_u_sphere_plotting[category]
    kwdict['azim'] = azimuth_angles[category]
    kwdict['color'] = plotting_color[category]
    return kwdict


def pc_sampler(original_incomplete_file, n_samples, save_file=None, dtype=np.float32):
    '''NEW - will be used to sample each file from Matthias and store them.
    '''
    pc = Point_Cloud(ply_file=original_incomplete_file)
    pc.permute_points([0, 2, 1])
    pc, _ = pc.sample(n_samples)
    pc.points = pc.points.astype(dtype)
    pc.lex_sort()
    if save_file is not None:
        pc.save_as_ply(save_file)
    return pc


def pc_loader(ply_file):
    vscan_scan_pattern = '__?__.ply'
    pc = Point_Cloud(ply_file=ply_file)
    tokens = ply_file.split('/')
    model_id = tokens[-1][:-len(vscan_scan_pattern)]
    scan_id = tokens[-1][-len(vscan_scan_pattern):-(len('.ply'))]
    syn_id = tokens[-2]
    return pc.points, (model_id, scan_id), syn_id


def load_partial_pointclouds(file_list, top_in_dir, n_threads, ending='.ply', class_restriction=None):
    file_names = []
    with open(file_list, 'r') as f_in:
        for line in f_in:
            syn_id, model_name, scan_id = line.split()
            if class_restriction is not None and syn_id not in class_restriction:
                continue
            else:
                file_names.append(osp.join(top_in_dir, syn_id, model_name + '__' + scan_id + '__' + ending))

    pclouds, model_ids, class_ids = load_crude_point_clouds(file_names=file_names, n_threads=n_threads, loader=pc_loader)
    print('{0} partial point clouds were loaded.'.format(len(pclouds)))
    return pclouds, model_ids, class_ids


def make_validation_from_train_data(train_data, validation_percent):
    all_tr_labels = np.array([x[:-6] for x in train_data.labels], dtype=object)  # Remove the scan_id.
    all_tr_labels_u = np.unique(all_tr_labels)
    train_new = train_validate_test_split(all_tr_labels_u, train_perc=1.0 - validation_percent, validate_perc=validation_percent, test_perc=0.0, shuffle=False)[0]
    tr_indicator = set(train_new)
    mask = np.zeros(len(all_tr_labels), dtype=np.bool)
    for i, label in enumerate(all_tr_labels):
        if label in tr_indicator:
            mask[i] = True

    train_data_ = PointCloudDataSet(train_data.point_clouds[mask], labels=train_data.labels[mask], noise=train_data.noisy_point_clouds[mask], init_shuffle=False)
    mask = np.logical_not(mask)
    val_data = PointCloudDataSet(train_data.point_clouds[mask], labels=train_data.labels[mask], noise=train_data.noisy_point_clouds[mask], init_shuffle=False)
    return train_data_, val_data


def permissible_dictionary(file_with_ids):
    ''' Returns a dictionary with model_ids that are white-listed in the input file.
    '''
    data_dict = defaultdict(dict)
    with open(file_with_ids, 'r') as f_in:
        for line in f_in:
            syn_id, model_id, scan_id = line.split()
            if len(data_dict[syn_id]) == 0:
                data_dict[syn_id] = set()
            data_dict[syn_id].add((model_id + scan_id))
    return data_dict


def mask_of_permissible(model_names, permissible_file, class_syn_id=None):
    ''' model_names: N x 1 np.array
        returns : mask N x 1 boolean'''
    permissible_dict = permissible_dictionary(permissible_file)
    if class_syn_id is not None:
        permissible_dict = permissible_dict[class_syn_id]

    mask = np.zeros([len(model_names)], dtype=np.bool)
    for i, model_id in enumerate(model_names):
        if model_id in permissible_dict:
            mask[i] = True

    return mask


def load_signed_distance_field(sdf_file_name):
    ''' Loads the signed_distance_field and the known-uknown space mask according to Dataset of Angela et al.
    '''
    with open(sdf_file_name, 'rb') as fin:
        _ = struct.unpack('f', fin.read(4))[0]
        _ = struct.unpack('f', fin.read(4))[0]
        _ = struct.unpack('f', fin.read(4))[0]

        output_width = struct.unpack('I', fin.read(4))[0]
        output_height = struct.unpack('I', fin.read(4))[0]
        output_depth = struct.unpack('I', fin.read(4))[0]

        n_values = output_width * output_height * output_height
        sdf_list = np.zeros(n_values, dtype=np.float32)
        weight_list = np.zeros(n_values, dtype=np.int)
        for i in xrange(n_values):
            sdf = struct.unpack('f', fin.read(4))[0]
            _ = struct.unpack('I', fin.read(4))[0]  # free counter (ignored)
            _ = struct.unpack('B' * 3, fin.read(3))  # color (ignored)
            weight = struct.unpack('B', fin.read(1))[0]
            sdf_list[i] = sdf
            weight_list[i] = weight

    sdf_values = np.zeros((output_width, output_height, output_depth), dtype=np.float32)
    unknown_space_mask = np.zeros((output_width, output_height, output_depth), dtype=np.bool)

    for d in xrange(output_depth):
        for w in xrange(output_width):
            for h in xrange(output_height):
                idx = h + (d * 32) + (w * 32 * 32)
                sdf_values[d, h, w] = sdf_list[idx]
                if sdf_list[idx] >= -1 and weight_list[idx] >= 1:    # Known space
                    pass
                else:
                    unknown_space_mask[d, h, w] = True

    return sdf_values, unknown_space_mask


def load_unsigned_distance_field(df_file_name, truncate_thres=None):
    with open(df_file_name, 'rb') as fin:
        _ = struct.unpack('f', fin.read(4))[0]
        _ = struct.unpack('f', fin.read(4))[0]
        _ = struct.unpack('f', fin.read(4))[0]

        output_width = struct.unpack('Q', fin.read(8))[0]
        output_height = struct.unpack('Q', fin.read(8))[0]
        output_depth = struct.unpack('Q', fin.read(8))[0]

        output_grid = np.ndarray((output_width, output_height, output_depth), np.float32)

        n_output = output_width * output_height * output_depth
        output_grid_values = struct.unpack('f' * n_output, fin.read(4 * n_output))

    k = 0
    for d in range(output_depth):
        for w in range(output_width):
            for h in range(output_height):
                output_grid[w, h, d] = output_grid_values[k]
                k += 1

    if truncate_thres is not None:
        output_grid[output_grid > truncate_thres] = truncate_thres   # if you are more than three voxels away from the surface.

    return output_grid


def export_distance_field_to_text(out_file, field_values):
    with open(out_file, 'w') as fout:
        for i in np.nditer(field_values):
            fout.write(str(i) + '\n')
