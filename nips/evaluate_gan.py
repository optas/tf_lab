'''
Created on April 26, 2017

@author: optas
'''

import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors


def compute_3D_grid(resolution=32):
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in xrange(resolution):
        for j in xrange(resolution):
            for k in xrange(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5
    return grid, spacing


def distance_field_from_nn(pointcloud, grid_resolution, k):
    r = grid_resolution
    grid, spacing = compute_3D_grid(r)
    grid = grid.reshape(-1, 3)
    nn = NearestNeighbors(n_neighbors=k).fit(pointcloud)
    distances, _ = nn.kneighbors(grid)
    distances = np.average(distances, axis=1)
    distances = distances.astype(np.float32)
    distances = distances.reshape(r, r, r, 1)
    distances /= spacing
    return distances


def entropy_of_occupancy_grid(pclouds, grid_resolution):
    grid_counters = np.zeros((grid_resolution, grid_resolution, grid_resolution)).reshape(-1)
    grid_coordinates, _ = compute_3D_grid(grid_resolution)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        indices = np.unique(indices)
        for i in indices:
            grid_counters[i] += 1

    acc_entropy = 0.0
    n = len(pclouds)
    assert(np.max(grid_counters) <= n)
    counter = 0.0
    for g in grid_counters:
        if g > 0:
            counter += 1
            p = g / float(n)
            acc_entropy += entropy([p, 1.0 - p])
    return acc_entropy / counter


def inception_like_score(classification_net, pclouds, class_id=None):
    soft_max_out = classification_net.lala(pclouds)
    if np.any(soft_max_out < 0):
        raise ValueError('Probabilities have to be non-negative.')
#     entropy(soft_max_out.T)?

#     entropy(empty_grid)?