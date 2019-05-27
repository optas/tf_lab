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


def euclidean_k_neighbors(n_dims, k):
    """
    Use with ```tf.make_template``` to have access to place-holders - operations to compute
    the Euclidean knn among features, that are (batch_size x n_dims).  
    """    
    A_pl = tf.placeholder(tf.float32, [None, n_dims], 'euclidean_k_neighbors_feats_a')
    B_pl = tf.placeholder(tf.float32, [None, n_dims], 'euclidean_k_neighbors_feats_b')        
    inner_prod = tf.matmul(A_pl, B_pl, transpose_b=True)
    a_norm_sq = tf.maximum(tf.reduce_sum(A_pl * A_pl, 1), 0.0)
    b_norm_sq = tf.maximum(tf.reduce_sum(B_pl * B_pl, 1), 0.0)
    euclid_sq_dist = tf.expand_dims(a_norm_sq, 1) - 2.0 * inner_prod + tf.expand_dims(b_norm_sq, 0)        
    euclid_sq_sim = tf.negative(tf.maximum(euclid_sq_dist, 0.0))    
    sims, indices = tf.nn.top_k(euclid_sq_sim, k=k)
    euclid_sq_dist = tf.negative(sims)
    return euclid_sq_dist, indices, A_pl, B_pl