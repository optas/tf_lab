'''
Created on April 26, 2017

@author: optas
'''

import numpy as np
import socket
import tensorflow as tf
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from . helper import compute_3D_grid

try:
    if socket.gethostname() == socket.gethostname() == 'oriong2.stanford.edu':
        from .. external.oriong2.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.oriong2.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
    else:
        from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded.')


def entropy_of_occupancy_grid(pclouds, grid_resolution):
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


def emd_distances(pclouds, batch_size):
    # TODO -> Jinwei
    # Iterate over all pairs + batch them.
    # Need tf session 
    # suggestion -> open it in a different function so we can use it also for Chamfer distances.
    pc_1 = None
    pc_2 = None
    match = approx_match(pc_1, pc_2)
    tf.reduce_mean(match_cost(pc_1, pc_2, match))


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
