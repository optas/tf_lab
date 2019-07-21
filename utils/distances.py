"""
High level functions that compute distances among features/vector spaces. 

Created on May 25, 2019

@author: optas
"""

import numpy as np
import tensorflow as tf
from general_tools import iterate_in_chunks


def compute_k_neighbors(a_feat, b_feat, sim_op, knn_op, a_pl, b_pl,  
                        sess, batch_size=512):
    """ Input:
              a_feat: (N1 x feat_dim) numpy array
              b_feat: (N2 x feat_dim) numpy array
              sim_op, knn_op: tensorflow operations for similarity and knn-ids
              a_pl, b_pl: tensorflow placeholders to feed the provided features.
        
        Example:
        sim_op, knn_op, a_pl, b_pl =  cosine_k_neighbors(4096, 10)
        x = np.random(100, 4096)
        y = np.random(100, 4096)
        compute_k_neighbors(x, y, sim_op, knn_op, a_pl, b_pl)               
    """
    
    n_a, n_dims = a_feat.shape
    _, dummy = b_feat.shape
    assert(n_dims == dummy)
    
    s = []  # similarities
    i = []  # neighbors
    
    for idx in iterate_in_chunks(np.arange(n_a), batch_size):        
        r = sess.run([sim_op, knn_op], feed_dict={a_pl:a_feat[idx], b_pl:b_feat})
        s.append(r[0])
        i.append(r[1])
    return np.vstack(s), np.vstack(i)


def cosine_k_neighbors(n_dims, k):
    """
    Use with ```tf.make_template``` to have access to place-holders - operations to compute
    the cosine knn among features, that are (batch_size x n_dims).  
    """
    A_pl = tf.placeholder(tf.float32, [None, n_dims], 'cosine_k_neighbors_feats_a')
    B_pl = tf.placeholder(tf.float32, [None, n_dims], 'cosine_k_neighbors_feats_b')
    a = tf.nn.l2_normalize(A_pl, -1)
    b = tf.nn.l2_normalize(B_pl, -1)
    cos_sim = tf.matmul(a, b, transpose_b=True)
    sims, indices = tf.nn.top_k(cos_sim, k=k)
    return sims, indices, A_pl, B_pl


def euclidean_k_neighbors_with_place_holders(n_dims, k):
    """
    Use with ```tf.make_template``` to have access to place-holders - operations to compute
    the Euclidean knn among features, that are (batch_size x n_dims).  
    """    
    pl_a = tf.placeholder(tf.float32, [None, n_dims], 'euclidean_k_neighbors_feats_a')
    pl_b = tf.placeholder(tf.float32, [None, n_dims], 'euclidean_k_neighbors_feats_b')        
    euclid_sq_dist, indices = euclidean_k_neighbors(pl_a, pl_b, k)
    return euclid_sq_dist, indices, pl_a, pl_b


def euclidean_k_neighbors(feat_a, feat_b, k, identities=None):
    """ Compute the euclidean k-nearest-neighbors of each feat_a among all feat_b.
        Args:
            feat_a (tf.Tensor): (Ma, N) matrix containing Ma, N-dimensional features.
            feat_b (tf.Tensor): (Mb, N) matrix containing Mb, N-dimensional features.
            k (int): number of nearest neighbors
	    identities (tf.Tensor): (Ma,) the distance (feat_a[m], feat_b[identity[m]]) will be maximal so to be excluded.
        Returns:
            euclid_sq_dist: (Ma x k) square-euclidean distances of knn.
            indices: (Ma x k)
    """
    inner_prod = tf.matmul(feat_a, feat_b, transpose_b=True)
    a_norm_sq = tf.maximum(tf.reduce_sum(feat_a * feat_a, 1), 0.0)
    b_norm_sq = tf.maximum(tf.reduce_sum(feat_b * feat_b, 1), 0.0)
    euclid_sq_dist = tf.expand_dims(a_norm_sq, 1) - 2.0 * inner_prod + tf.expand_dims(b_norm_sq, 0)
    if identities is not None:
        batch_size = tf.shape(identities)[0]
        indices = tf.stack([tf.range(batch_size), identities], 1)
        updates = tf.reduce_max(euclid_sq_dist) + 1
        updates = tf.tile(tf.expand_dims(updates, -1), [batch_size])
        shape = tf.shape(euclid_sq_dist)
        canceling = tf.scatter_nd(indices, updates, shape)
        euclid_sq_dist += canceling 
    euclid_sq_sim = tf.negative(tf.maximum(euclid_sq_dist, 0.0))
    sims, indices = tf.nn.top_k(euclid_sq_sim, k=k)
    euclid_sq_dist = tf.negative(sims)
    return euclid_sq_dist, indices
