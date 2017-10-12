'''
Created on October 11, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np
import warnings
import os.path as osp
import time

from scipy.stats import entropy
from general_tools.simpletons import iterate_in_chunks
from .. nips.helper import compute_3D_grid, compute_3D_sphere

try:
    from sklearn.neighbors import NearestNeighbors
except:
    print ('sklearn module not installed.')

try:
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
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        warnings.warn('Point-clouds not in in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds not unit sphere.')

    if in_sphere:
        grid_coordinates, _ = compute_3D_sphere(grid_resolution)
    else:
        grid_coordinates, _ = compute_3D_grid(grid_resolution)

    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

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
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)      # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    '''another way of computing JSD'''
    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


def sample_pclouds_distances(pclouds, batch_size, n_samples, dist='emd', sess=None):
    '''
        pclouds: numpy array (n_pc * n_points * 3)
    '''
    if np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > 0.5 + 10e-4:
        raise ValueError('Point-clouds have to be in unit sphere.')

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
        loss_list.append(loss_d[0])

    return loss_list


def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=False, sess=None, verbose=False):
    ''' normalize (boolean): if True the Chamfer distance between two point-clouds is the average of matched
                             point-distances. Alternatively, is their sum.
    '''
    if normalize:
        reducer = tf.reduce_mean
    else:
        reducer = tf.reduce_sum

    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible Point-Clouds.')

    # TF Graph Operations
    ref_pl = tf.placeholder(tf.float32, shape=(1, n_pc_points, pc_dim))
    sample_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points, pc_dim))

    repeat_times = tf.shape(sample_pl)[0]   # slower- could be used to use entire set of samples.
#     repeat_times = batch_size
    ref_repeat = tf.tile(ref_pl, [repeat_times, 1, 1])
    ref_repeat = tf.reshape(ref_repeat, [repeat_times, n_pc_points, pc_dim])

    ref_to_s, _, s_to_ref, _ = nn_distance(ref_repeat, sample_pl)
    chamfer_dist_batch = reducer(ref_to_s, 1) + reducer(s_to_ref, 1)

    best_in_batch = tf.reduce_min(chamfer_dist_batch)   # Best distance, of those that were matched to single ref pc.
    matched_dists = []
    for i in xrange(n_ref):
        best_in_all_batches = []
        if verbose and i % 50 == 0:
            print i
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            if len(sample_chunk) != batch_size:
                continue
            feed_dict = {ref_pl: np.expand_dims(ref_pcs[i], 0), sample_pl: sample_chunk}
            b = sess.run(best_in_batch, feed_dict=feed_dict)
            best_in_all_batches.append(b)

        matched_dists.append(np.min(best_in_all_batches))

    mmd = np.mean(matched_dists)
    return mmd, matched_dists
