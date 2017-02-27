'''
Created on Feb 27, 2017

@author: optas
'''

import time
import numpy as np
from collections import defaultdict
from geo_tool import Point_Cloud 

from .. point_clouds.in_out import load_filenames_of_input_data, match_incomplete_to_complete_data, load_crude_point_clouds, PointCloudDataSet

vscan_search_pattern = '.ply'
vscan_scan_pattern = '__?__.ply'
_n_samples = 2048


def permissible_models(file_with_ids):
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


def load_permissible_pointclouds(v_scan_data_top_dir, permissible_dict, n_threads):
    noisy_pcloud_files = load_filenames_of_input_data(v_scan_data_top_dir, vscan_search_pattern)
    keep = np.zeros([len(noisy_pcloud_files)], dtype=np.bool)
    for i, f in enumerate(noisy_pcloud_files):
        model_id = f.split('/')[-1][:-len(vscan_scan_pattern)]
        if model_id in permissible_dict:
            keep[i] = True

    noisy_pcloud_files = np.array(noisy_pcloud_files, dtype=object)
    noisy_pcloud_files = noisy_pcloud_files[keep]
    noisy_pclouds, noisy_ids, class_ids = load_crude_point_clouds(file_names=noisy_pcloud_files, n_threads=n_threads, loader=_load_virtual_scan_incomplete_pcloud)
    print '%d files containing noisy point clouds were found.' % (len(noisy_pclouds), )
    return noisy_pclouds, noisy_ids, class_ids


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


def load_single_class_data(top_data_dir, permissible_file_list, class_syn_id, full_pclouds, full_model_names, n_threads, n_samples):
    # Currently works with single class.
    global n_samples_
    n_samples_ = n_samples
    data_dict = permissible_models(permissible_file_list)
    data_dict = data_dict[class_syn_id]
    incomplete_pclouds, initial_ids, _ = load_permissible_pointclouds(top_data_dir, data_dict, n_threads)
    full_pclouds_matched, ids = match_to_complete_data(initial_ids, full_model_names, full_pclouds)
    return PointCloudDataSet(full_pclouds_matched, noise=incomplete_pclouds, labels=ids)
