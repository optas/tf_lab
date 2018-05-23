'''
Created on May 21, 2018

@author: https://github.com/WangYueFt/dgcnn

'''
import tensorflow as tf


def pairwise_distance(point_cloud):
    '''Compute euclidean pairwise distance between all points of a point cloud.
        Trick: |a_1 - a_2|^2 = |a_1|^2 + |a_2|^2 - 2 a b
    Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
    pairwise distance: (batch_size, num_points, num_points)
    '''
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)           # All a.innerproduct(b) terms.
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True)  # Keepdims=True to do broadcasting and add it in all rows.
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
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
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
