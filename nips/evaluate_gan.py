'''
Created on April 26, 2017

@author: optas, jingweij
'''


import socket
import warnings
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

from . helper import compute_3D_grid, compute_3D_sphere


try:
    if socket.gethostname() == socket.gethostname() == 'oriong2.stanford.edu':
        from .. external.oriong2.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.oriong2.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
    else:
        from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded.')


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    if abs(np.max(pclouds)) > 0.5 or abs(np.min(pclouds)) > 0.5:
        raise ValueError('Point-clouds are expected to be in unit cube.')

    grid_counters = np.zeros((grid_resolution, grid_resolution, grid_resolution)).reshape(-1)
    grid_bernoulli_rvars = np.zeros((grid_resolution, grid_resolution, grid_resolution)).reshape(-1)

    if in_sphere:
        grid_coordinates, _ = compute_3D_sphere(grid_resolution)
    else:
        grid_coordinates, _ = compute_3D_grid(grid_resolution)

    grid_coordinates = grid_coordinates.reshape(-1, 3)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=5).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    '''
    TODO: move to general tools
    '''
    M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, M) + entropy(Q, M))


def point_cloud_distances(pclouds, block_size, dist='emd', sess=None):
    ''' pclouds: numpy array (n_pc * n_points * 3)
        block_size: int: pairwise distances among these many point-clouds will be computes.
    '''

    if abs(np.max(pclouds)) > 0.5 or abs(np.min(pclouds)) > 0.5:
        warnings.warn('Point-clouds are not expected to be in unit cube.')

    num_clouds, num_points, dim = pclouds.shape
    batch_size = block_size * (block_size - 1)
    num_units = num_clouds // block_size
    pc_1_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, dim))
    pc_2_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, dim))

    if sess is None:
        sess = tf.Session()

    dist = dist.lower()
    if dist == 'emd':
        match = approx_match(pc_1_pl, pc_2_pl)
        loss = tf.reduce_mean(match_cost(pc_1_pl, pc_2_pl, match))
    elif dist == 'chamfer':
        cost_p1_p2, _, cost_p2_p1, _ = nn_distance(pc_1_pl, pc_2_pl)
        loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
    else:
        raise ValueError()

    loss_list = []
    for u in range(num_units):
        pc_idx = np.arange(u * block_size, (u + 1) * block_size)
        pc_idx1 = np.repeat(pc_idx, block_size - 1)
        pc_idx2 = np.tile(pc_idx, block_size)
        mask = np.arange(block_size ** 2)
        mask = mask[mask % (block_size + 1) != 0]
        pc_idx2 = pc_idx2[mask]
        pc1 = pclouds[pc_idx1, :, :]
        pc2 = pclouds[pc_idx2, :, :]
        loss_d = sess.run([loss], feed_dict={pc_1_pl: pc1, pc_2_pl: pc2})
        loss_list.append(loss_d[0] * batch_size)
    return loss_list


def sample_pclouds_distances(pclouds, batch_size, n_samples, dist='emd', sess=None):
    num_clouds, num_points, dim = pclouds.shape
    pc_1_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, dim))
    pc_2_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, dim))

    dist = dist.lower()
    if dist == 'emd':
        match = approx_match(pc_1_pl, pc_2_pl)
        loss = tf.reduce_mean(match_cost(pc_1_pl, pc_2_pl, match))
    elif dist == 'chamfer':
        cost_p1_p2, _, cost_p2_p1, _ = nn_distance(pc_1_pl, pc_2_pl)
        loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
    else:
        raise ValueError()

    if sess is None:
        sess = tf.Session()

    loss_list = []
    for _ in range(n_samples):
        rids = np.random.choice(range(num_clouds), batch_size * 2, replace=False)
        pc_idx1 = rids[:len(rids) // 2]
        pc_idx2 = rids[len(rids) // 2:]
        pc1 = pclouds[pc_idx1, :, :]
        pc2 = pclouds[pc_idx2, :, :]
        loss_d = sess.run([loss], feed_dict={pc_1_pl: pc1, pc_2_pl: pc2})
        loss_list.append(loss_d[0] * batch_size)

    return loss_list


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
