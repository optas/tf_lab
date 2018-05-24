'''
Created on May 21, 2018

@authors: https://github.com/WangYueFt/dgcnn, optas

'''
import tensorflow as tf
import ipdb
from .. fundamentals.utils import safe_norm

def pairwise_angles(layer, epsilon=10e-7):
    ''' layer: B x N x D
        output: B x N x N angles stemmed from each pair in the D-dimensional feature space.
    '''
    inn_prod = tf.matmul(layer, layer, transpose_b=True)  #inner-product
    row_norm = safe_norm(layer, axis=-1, epsilon=1e-7, keep_dims=True)
    col_norm = tf.transpose(row_norm, perm=[0, 2, 1])
    pw_cosine = inn_prod / row_norm
    pw_cosine = pw_cosine / col_norm
    pw_angle = tf.acos(pw_cosine)
    return pw_cosine


def soft_maxed_edge(layer, aggregation='sum'):
    '''First attempt to imitate capsules'''
    A = pairwise_angles(layer)
    feat_sim = tf.nn.softmax(A, dim=2, name='soft_max_angles')
    if aggregation == 'sum': 
        res = tf.matmul(feat_sim, layer, name='sum_aggregate')
    else:
        raise NotImplemented()
    return res


def pairwise_distance(point_cloud):
    '''Compute euclidean pairwise distance between all points of a point cloud.
        Trick: |a_1 - a_2|^2 = |a_1|^2 + |a_2|^2 - 2 a b
    Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
    pairwise distance: (batch_size, num_points, num_points)
    '''
    og_batch_size = point_cloud.get_shape().as_list()[0]
    #point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)           # All a.innerproduct(b) terms.
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)  # Keepdims=True to do broadcasting and add it in all rows.
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])      # Add in all columns.
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    '''Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (batch_size, num_points, num_points)
        k: int

    Returns:
        nearest neighbors: (batch_size, num_points, k)
    Note: it also returns the id of the "self" of each point, if diagonal of adj_matrix is 0.
    '''
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20):
    '''Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
    edge features: (batch_size, num_points, k, num_dims)
    '''
    og_batch_size = point_cloud.get_shape().as_list()[0]        
    #point_cloud = tf.squeeze(point_cloud)
    
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()    
    #batch_size = point_cloud_shape[0].value
    batch_size = tf.shape(point_cloud)[0]
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)  # Add absolute and relative features.    
    return edge_feature