'''
Created on April 26, 2017

@author: optas
'''

import numpy as np
import socket
import tensorflow as tf
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import os
import sys
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_ops'))
#from IPython import embed; embed()
#import build_assignment
#import hungarian_match
#import hungarian_match_cost

try:
    if socket.gethostname() == socket.gethostname() == 'oriong2.stanford.edu' or \
       socket.gethostname() == socket.gethostname() == 'oriong3.stanford.edu':
        print "here"
        from .. external.oriong2.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.oriong2.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
    else:
        from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded.')


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


def emd_distances(pclouds, unit_size, sess):
    # author: Jingwei Ji
    # Iterate over all pairs + batch them.
    # Need tf session 
    # suggestion -> open it in a different function so we can use it also for Chamfer distances.
    # pclouds: (num_clouds, num_points, 3)
    # every batch evaluate (unit_size * (unit_size - 1)) pairs of pclouds
    # TODO: fix import problem
    import os
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_dir, '../external/Chamfer_EMD_losses'))
    from tf_nndistance import nn_distance
    from tf_approxmatch import approx_match, match_cost
    from scipy.spatial.distance import cdist

    num_clouds, num_points, dim = pclouds.shape
    batch_size = unit_size * (unit_size - 1)
    num_units = num_clouds // unit_size
    pc_1_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, dim))
    pc_2_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, dim))
    match = approx_match(pc_1_pl, pc_2_pl)
    mat_loss = match_cost(pc_1_pl, pc_2_pl, match)
    loss = tf.reduce_mean(match_cost(pc_1_pl, pc_2_pl, match))

    loss_list = []
    for u in range(num_units):
        pc_idx = np.arange(u * unit_size, (u+1) * unit_size)
        pc_idx1 = np.repeat(pc_idx, unit_size - 1)
        pc_idx2 = np.tile(pc_idx, unit_size)
        mask = np.arange(unit_size ** 2)
        mask = mask[mask % (unit_size+1) != 0]
        pc_idx2 = pc_idx2[mask]
        pc1 = pclouds[pc_idx1, :, :]
        pc2 = pclouds[pc_idx2, :, :]

        loss_d, mat_loss_d = sess.run([loss, mat_loss], feed_dict={pc_1_pl:pc1, pc_2_pl:pc2})
        loss_list.append(loss_d)
    return np.mean(loss_list)

if __name__ == "__main__":
    unit_size = 2
    sess = tf.Session()
    data_path = '/orions4-zfs/projects/lins2/Panos_Space/DATA/NIPS/our_samples/gt_chair.npz'
    data = np.load(data_path)['arr_0']
    dist = emd_distances(data[:50,...], 5, sess)
    dist2 = emd_distances(data[:50,...], 50, sess)
    from pdb import set_trace; set_trace()



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
