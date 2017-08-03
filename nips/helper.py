'''
Created on May 3, 2017

@author: optas
'''

import numpy as np
from numpy.linalg import norm
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geo_tool import Point_Cloud

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


def plot_probability_space_on_voxels(voxel_resolution, prb_thres, three_d_variable, in_sphere=True):
    ''' Used to visualize JSD measurements.
        prb_thres: [0,1] float, only plot cells that have higher than prb_thres mass.
    '''
    if in_sphere:
        grid_centers, _ = compute_3D_sphere(voxel_resolution)
    else:
        grid_centers, _ = compute_3D_grid(voxel_resolution)

    grid_centers = grid_centers.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    three_d_variable /= np.sum(three_d_variable)
    grid_centers = grid_centers[three_d_variable > prb_thres]
    c = three_d_variable[three_d_variable > prb_thres].tolist()

    ax.scatter(grid_centers[:, 0], grid_centers[:, 1], grid_centers[:, 2],
               marker='s', s=10, c=c, cmap='jet')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    return fig
