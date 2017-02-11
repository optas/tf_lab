'''
Created on February 10, 2017

'''

import tensorflow as tf
from tflearn.layers.conv import conv_1d, conv_3d_transpose
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import batch_normalization

# if spn:
#     with tf.variable_scope('transform_input') as sc:
#         transform = input_transform_net(point_cloud,batch_size)
#     layer = tf.batch_matmul(layer,transform)
        
def transformer(poin_cloud, batch_size, K=3):
    """ Input (XYZ) Transform Net, input is B x N x 3 gray image
    Return:
        Transformation matrix of size 3 x K """
    num_point = 1024

    net = conv_1d(input_image, 64, 1, 1)
    net = batch_normalization(net)
    net = tf.nn.relu(net)

    net = conv_1d(input_image, 128, 1, 1)
    net = batch_normalization(net)
    net = tf.nn.relu(net)

    net = conv_1d(input_image, 1024, 1, 1)
    net = batch_normalization(net)
    net = tf.nn.relu(net)

    net = tf.reduce_max(net, 1)
    net = fully_connected(layer, 512, activation='relu', weights_init='xavier')
    net = fully_connected(layer, 256, activation='relu', weights_init='xavier')

    with tf.variable_scope('transform_XYZ'):
        weights = tf.get_variable('weights', [256, 3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [-1, 3, K])
    return transform
