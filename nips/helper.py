'''
Created on May 3, 2017

@author: optas
'''

import numpy as np
from geo_tool import Point_Cloud
from numpy.linalg import norm
import os.path as osp

from .. point_clouds.in_out import load_filenames_of_input_data, load_crude_point_clouds
from . data_sets.shape_net import shape_net_category_to_synth_id
from . data_sets.shape_net import pc_loader as sn_pc_loader


def load_shape_net_models_used_by_wu(n_pc_samples, pclouds_path):
    wu_cat_names, wu_syn_ids = wu_nips_16_categories()

    pclouds = []
    model_ids = []
    syn_ids = []

    for cat_name, syn_id in zip(wu_cat_names, wu_syn_ids):
        print cat_name, syn_id
        file_names = load_filenames_of_input_data(osp.join(pclouds_path, syn_id), '.ply')
        pclouds_temp, model_ids_temp, syn_ids_temp = load_crude_point_clouds(file_names=file_names, n_threads=50, loader=sn_pc_loader)
        print '%d files containing complete point clouds were found.' % (len(pclouds_temp), )
        pclouds.append(pclouds_temp)
        model_ids.append(model_ids_temp)
        syn_ids.append(syn_ids_temp)

    pclouds = np.vstack(pclouds)
    model_ids = np.hstack(model_ids)
    syn_ids = np.hstack(syn_ids)
    return pclouds, model_ids, syn_ids


def average_per_class(lsvc, test_emb, gt_labels):
    y_pred = lsvc.predict(test_emb)
    gt_labels = np.array(gt_labels)
    scores_per_class = []

    for c in np.unique(gt_labels):
        index_c = gt_labels == c
        n_class = float(np.sum(index_c))
        s = np.sum(gt_labels[index_c] == y_pred[index_c]) 
        s /= n_class
        scores_per_class.append(s)
    return np.mean(scores_per_class)


def compute_3D_grid(resolution):
    '''Returns the center coordinates of each cell of a 3D Grid with resolution^3 cells.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in xrange(resolution):
        for j in xrange(resolution):
            for k in xrange(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5
    return grid, spacing


def compute_3D_sphere(resolution):
    grid, spacing = compute_3D_grid(resolution=resolution)
    pts = grid.reshape(-1, 3)
    pts = pts[norm(pts, axis=1) <= 0.5]  # clip in half-sphere
    print len(pts)
#     pc = Point_Cloud(pts).center_in_unit_sphere()
#     pts = pc.points
    return pts, spacing


def zero_mean_half_sphere(in_pclouds):
    ''' Zero MEAN + Max_dist = 0.5
    '''
    pclouds = in_pclouds.copy()
    pclouds = pclouds - np.expand_dims(np.mean(pclouds, axis=1), 1)
    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    dist = np.expand_dims(np.expand_dims(dist, 1), 2)
    pclouds = pclouds / (dist * 2.0)
    return pclouds


def visualize_voxel_content_as_pcloud(voxel_data):
    x, y, z = np.where(voxel_data)
    points = np.vstack((x, y, z)).T
    return Point_Cloud(points=points).plot()


def pclouds_centered_and_half_sphere(pclouds):
    for i, pc in enumerate(pclouds):
        pc, _ = Point_Cloud(pc).center_axis()
        pclouds[i] = pc.points

    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    dist = np.expand_dims(np.expand_dims(dist, 1), 2)
    pclouds = pclouds / (dist * 2.0)

    return pclouds


def wu_nips_16_categories():
    category_names = ['airplane', 'car', 'chair', 'sofa', 'rifle', 'boat', 'table']
    syn_id_dict = shape_net_category_to_synth_id()
    return category_names, [syn_id_dict[i] for i in category_names]


def center_pclouds_in_unit_sphere(pclouds):
    for i, pc in enumerate(pclouds):
        pc, _ = Point_Cloud(pc).center_axis()
        pclouds[i] = pc.points

    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    dist = np.expand_dims(np.expand_dims(dist, 1), 2)
    pclouds = pclouds / (dist * 2.0)

    for i, pc in enumerate(pclouds):
        pc, _ = Point_Cloud(pc).center_axis()
        pclouds[i] = pc.points

    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    assert(np.all(abs(dist - 0.5) < 0.0001))
    return pclouds
