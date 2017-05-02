'''
Created on April 26, 2017

@author: optas
'''

import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors


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


def entropy_of_occupancy_grid_(pclouds, grid_resolution):
    '''
    Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
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
    n = float(len(pclouds))
    for g in grid_counters:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])
    return acc_entropy / len(grid_counters)


def classification_confidences(classification_net, pclouds, class_id):
    '''
    Inputs:
        classification_net:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        class_id: (int) index corresponding to the 'target' class in the classsification_net
    '''
    soft_max_out = classification_net.inference(pclouds)

    if np.any(soft_max_out < 0):
        raise ValueError('Probabilities have to be non-negative.')

    return soft_max_out[:, class_id]
