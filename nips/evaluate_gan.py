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
    empty_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution, 3)).reshape(-1, 3)
    grid_coordinates, _ = compute_3D_grid(grid_resolution)
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates.reshape(-1, 3))
    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        empty_grid[indices] += 1
    return entropy(empty_grid)


def inception_like_score(classification_net, pclouds):
    soft_max_out = classification_net.lala(pclouds)
    if np.any(soft_max_out < 0):
        raise ValueError('Probabilities have to be non-negative.')
#     entropy(soft_max_out.T)?

    
#     entropy(empty_grid)?