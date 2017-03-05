'''
Created on Feb 27, 2017

@author: optas
'''

import struct
import numpy as np
from collections import defaultdict
from geo_tool import Point_Cloud 

from ... point_clouds.in_out import load_filenames_of_input_data, load_crude_point_clouds, PointCloudDataSet
from .helper import match_incomplete_to_complete_data

vscan_search_pattern = '.ply'      # TODO use for both a regex.
vscan_scan_pattern = '__?__.ply'    # Used only to indicate how to crop the input file-names to produced model_ids etc.
_n_samples = 2048


def permissible_dictionary(file_with_ids):
    ''' Returns a dictionary with model_ids that are white-listed in the input file.
    '''
    data_dict = defaultdict(dict)
    with open(file_with_ids, 'r') as f_in:
        for line in f_in:
            syn_id, model_id = line.split()
            if len(data_dict[syn_id]) == 0:
                data_dict[syn_id] = set()
            data_dict[syn_id].add(model_id)
    return data_dict


def _load_virtual_scan_incomplete_pcloud(f_name):
    pc = Point_Cloud(ply_file=f_name)
    pc.permute_points([0, 2, 1])
    pc, _ = pc.sample(_n_samples)
    pc.lex_sort()
    pc.center_in_unit_sphere()
    tokens = f_name.split('/')
    model_id = tokens[-1][:-len(vscan_scan_pattern)]
    scan_id = tokens[-1][-len(vscan_scan_pattern):-(len('.ply'))]
    class_id = tokens[-2]
    return pc.points, (model_id, scan_id), class_id


def load_incomplete_pointclouds(v_scan_data_top_dir, permissible_dict, n_threads, search_pattern=vscan_search_pattern):
    incommplete_pcloud_files = load_filenames_of_input_data(v_scan_data_top_dir, search_pattern)
    keep = np.zeros([len(incommplete_pcloud_files)], dtype=np.bool)
    for i, f in enumerate(incommplete_pcloud_files):
        model_id = f.split('/')[-1][:-len(vscan_scan_pattern)]
        if model_id in permissible_dict:
            keep[i] = True

    incommplete_pcloud_files = np.array(incommplete_pcloud_files, dtype=object)
    incommplete_pcloud_files = incommplete_pcloud_files[keep]
    inc_pclouds, inc_ids, class_ids = load_crude_point_clouds(file_names=incommplete_pcloud_files, n_threads=n_threads, loader=_load_virtual_scan_incomplete_pcloud)
    print '%d incomplete point clouds were loaded.' % (len(inc_pclouds), )
    return inc_pclouds, inc_ids, class_ids


def match_to_complete_data(initial_ids, full_model_names, full_pclouds):
    scan_ids = np.empty([len(initial_ids)], dtype=object)
    incomplete_model_names = np.empty([len(initial_ids)], dtype=object)
    for i, item in enumerate(initial_ids):
        incomplete_model_names[i] = item[0]
        scan_ids[i] = item[1]

    # # Match incomplete to complete data.
    mapping = match_incomplete_to_complete_data(full_model_names, incomplete_model_names)
    full_pclouds_matched = full_pclouds[mapping]
    ids = incomplete_model_names + '.' + scan_ids
    return full_pclouds_matched, ids


def load_single_class_incomplete_dataset(top_data_dir, permissible_file_list, class_syn_id, full_pclouds, full_model_names, n_threads, n_samples, search_pattern=vscan_search_pattern):
    # Currently works with single class.
    global _n_samples
    _n_samples = n_samples
    data_dict = permissible_dictionary(permissible_file_list)
    data_dict = data_dict[class_syn_id]
    incomplete_pclouds, initial_ids, _ = load_incomplete_pointclouds(top_data_dir, data_dict, n_threads, search_pattern)
    full_pclouds_matched, ids = match_to_complete_data(initial_ids, full_model_names, full_pclouds)
    return PointCloudDataSet(full_pclouds_matched, noise=incomplete_pclouds, labels=ids)


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
        output_grid[output_grid > truncate_thres] = truncate_thres

    return output_grid


def export_distance_field_to_text(out_file, field_values):
    with open(out_file, 'w') as fout:
        for i in np.nditer(field_values):
            fout.write(str(i) + '\n')
