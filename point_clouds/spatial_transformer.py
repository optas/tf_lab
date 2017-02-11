'''
Created on February 10, 2017

'''

import tensorflow as tf
from tflearn.layers.conv import conv_1d, conv_3d_transpose
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import batch_normalization


def transformer(point_cloud):
    """ Input (XYZ) Transform Net, input is B x N x 3 gray image
    Return:
        Transformation matrix of size 3 x 3 """
    net = conv_1d(point_cloud, 64, 1, 1)
    net = batch_normalization(net)
    net = tf.nn.relu(net)

    net = conv_1d(net, 128, 1, 1)
    net = batch_normalization(net)
    net = tf.nn.relu(net)

    net = conv_1d(net, 1024, 1, 1)
    net = batch_normalization(net)
    net = tf.nn.relu(net)

    net = tf.reduce_max(net, 1)
    net = fully_connected(net, 512, activation='relu', weights_init='xavier')
    net = fully_connected(net, 256, activation='relu', weights_init='xavier')

    with tf.variable_scope('transform_XYZ'):
        weights = tf.get_variable('weights', [256, 3 * 3], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3 * 3], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [-1, 3, 3])
    return transform
