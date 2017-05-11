'''
Created on May 3, 2017

@author: optas
'''

import numpy as np
from geo_tool import Point_Cloud
from numpy.linalg import norm


def wu_nips_16_categoris():
    ['airplane', 'car', 'chair', 'sofa', 'rifle', 'boat', 'table']


def compute_3D_grid(resolution=32):
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


def compute_3D_sphere(resolution=32):
    grid, spacing = compute_3D_grid(resolution=32)
    pts = grid.reshape(-1, 3)
#     pts = pts[norm(pts, axis=1) <= 0.5]  # clip in half-sphere
    pc = Point_Cloud(pts).center_in_unit_sphere()
    pts = pc.points
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
