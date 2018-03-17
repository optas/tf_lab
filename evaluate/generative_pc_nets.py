'''
Created on October 11, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np
import warnings

from scipy.stats import entropy
from general_tools.simpletons import iterate_in_chunks
from .. nips.helper import compute_3D_grid, compute_3D_sphere

try:
    from sklearn.neighbors import NearestNeighbors
except:
    print ('sklearn module not installed.')

from .. external.structural_pc_losses import losses
nn_distance, approx_match, match_cost = losses()


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
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

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


def minimum_mathing_distance_tf_graph(n_pc_points, batch_size=None, normalize=False, sess=None, verbose=False, use_sqrt=False, use_EMD=False):
    ''' Produces the graph operations necessary to compute the MMD and consequently also the Coverage due to their 'symmetric' nature.
    Assuming a "reference" and a "sample" set of point-clouds that will be matched, this function creates the operation that matches
    a _single_ "reference" point-cloud to all the "sample" point-clouds given in a batch. Thus, is the building block of the function
    ```minimum_mathing_distance`` and ```coverage``` that iterate over the "sample" batches and each "reference" point-cloud.

    Args:
        n_pc_points (int): how many points each point-cloud of those to be compared has.
        batch_size (optional, int): if the iterator code that uses this function will
            use a constant batch size for iterating the sample point-clouds you can
            specify it hear to speed up the compute. Alternatively, the code is adapted
            to read the batch size dynamically.
        normalize (boolean): When the matching is based on Chamfer (default behavior), if True,
            the Chamfer is computed as the average of the matched point-wise squared euclidean
            distances. Alternatively, is their sum.
        use_sqrt (boolean): When the matching is based on Chamfer (default behavior), if True,
            the Chamfer is computed based on the (not-squared) euclidean distances of the
            matched point-wise euclidean distances.
        use_EMD (boolean): If true, the matchings are based on the EMD.
    '''
    if normalize:
        reducer = tf.reduce_mean
    else:
        reducer = tf.reduce_sum

    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    # Placeholders for the point-clouds: 1 for the reference (usually Ground-truth) and one of variable size for the collection
    # which is going to be matched with the reference.
    ref_pl = tf.placeholder(tf.float32, shape=(1, n_pc_points, 3))
    sample_pl = tf.placeholder(tf.float32, shape=(batch_size, n_pc_points, 3))

    if batch_size is None:
        batch_size = tf.shape(sample_pl)[0]

    ref_repeat = tf.tile(ref_pl, [batch_size, 1, 1])
    ref_repeat = tf.reshape(ref_repeat, [batch_size, n_pc_points, 3])

    if use_EMD:
        match = approx_match(ref_repeat, sample_pl)
        all_dist_in_batch = match_cost(ref_repeat, sample_pl, match)
    else:
        ref_to_s, _, s_to_ref, _ = nn_distance(ref_repeat, sample_pl)
        if use_sqrt:
            ref_to_s = tf.sqrt(ref_to_s)
            s_to_ref = tf.sqrt(s_to_ref)
        all_dist_in_batch = reducer(ref_to_s, 1) + reducer(s_to_ref, 1)

    best_in_batch = tf.reduce_min(all_dist_in_batch)   # Best distance, of those that were matched to single ref pc.
    location_of_best = tf.argmin(all_dist_in_batch, axis=0)
    return ref_pl, sample_pl, best_in_batch, location_of_best, sess


def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=False, sess=None, verbose=False, use_sqrt=False, use_EMD=False):
    '''Computes the MMD between two sets of point-clouds.

    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points that will be matched and
            compared to a set of "reference" point-clouds.
        ref_pcs (numpy array RxKx3): the R point-clouds, each of K points that constitute the set of
            "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to make
            the comparisons of the sample-vs-ref point-clouds.
        normalize (boolean): When the matching is based on Chamfer (default behavior), if True, the
            Chamfer is computed as the average of the matched point-wise squared euclidean distances.
            Alternatively, is their sum.
        use_sqrt: (boolean): When the matching is based on Chamfer (default behavior), if True, the
            Chamfer is computed based on the (not-squared) euclidean distances of the matched point-wise
             euclidean distances.
        sess (tf.Session, default None): if None, it will make a new Session for this.
        use_EMD (boolean: If true, the matchings are based on the EMD.

    Returns:
        A tuple containing the MMD and all the matched distances of which the MMD is their mean.
    '''

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')

    ref_pl, sample_pl, best_in_batch, _, sess = minimum_mathing_distance_tf_graph(n_pc_points, batch_size,
                                                                                  normalize=normalize, sess=sess,
                                                                                  use_sqrt=use_sqrt, use_EMD=use_EMD)
    matched_dists = []
    for i in xrange(n_ref):
        best_in_all_batches = []
        if verbose and i % 50 == 0:
            print i
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            feed_dict = {ref_pl: np.expand_dims(ref_pcs[i], 0), sample_pl: sample_chunk}
            b = sess.run(best_in_batch, feed_dict=feed_dict)
            best_in_all_batches.append(b)
        matched_dists.append(np.min(best_in_all_batches))
    mmd = np.mean(matched_dists)
    return mmd, matched_dists


def coverage(sample_pcs, ref_pcs, batch_size, normalize=False, sess=None, verbose=False, use_sqrt=False, use_EMD=False, ret_dist=False):
    '''Computes the Coverage between two sets of point-clouds.

    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points that will be matched
            and compared to a set of "reference" point-clouds.
        ref_pcs    (numpy array RxKx3): the R point-clouds, each of K points that constitute the
            set of "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to
            make the comparisons of the sample-vs-ref point-clouds.
        normalize  (boolean): When the matching is based on Chamfer (default behavior), if True,
            the Chamfer is computed as the average of the matched point-wise squared euclidean
            distances. Alternatively, is their sum.
        use_sqrt  (boolean): When the matching is based on Chamfer (default behavior), if True,
            the Chamfer is computed based on the (not-squared) euclidean distances of the matched
            point-wise euclidean distances.
        sess (tf.Session):  If None, it will make a new Session for this.
        use_EMD (boolean): If true, the matchings are based on the EMD.
        ret_dist (boolean): If true, it will also return the MMD distances of all the sample_pcs
            wrt. the ref_pcs.
        Returns: the coverage score and optionally the MMD distances of the samples_pcs.
    '''
    _, n_pc_points, pc_dim = ref_pcs.shape
    n_sam, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible Point-Clouds.')

    ref_pl, sample_pl, best_in_batch, loc_of_best, sess = minimum_mathing_distance_tf_graph(n_pc_points, batch_size,
                                                                                            normalize=normalize, sess=sess,
                                                                                            use_sqrt=use_sqrt, use_EMD=use_EMD)
    matched_gt = []
    matched_dist = []
    for i in xrange(n_sam):
        best_in_all_batches = []
        loc_in_all_batches = []

        if verbose and i % 50 == 0:
            print i

        for ref_chunk in iterate_in_chunks(ref_pcs, batch_size):
            feed_dict = {ref_pl: np.expand_dims(sample_pcs[i], 0), sample_pl: ref_chunk}
            b, loc = sess.run([best_in_batch, loc_of_best], feed_dict=feed_dict)
            best_in_all_batches.append(b)
            loc_in_all_batches.append(loc)

        best_in_all_batches = np.array(best_in_all_batches)
        b_hit = np.argmin(best_in_all_batches)    # In which batch it minimum occurred.
        matched_dist.append(np.min(best_in_all_batches))
        hit = np.array(loc_in_all_batches)[b_hit]
        matched_gt.append(batch_size * b_hit + hit)
    if ret_dist:
        return matched_gt, matched_dist
    else:
        return matched_gt
