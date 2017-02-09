'''
Created on February 4, 2017

@author: optas
'''

import tensorflow as tf

try:
    from tflearn.layers.conv import conv_3d_transpose
except:
    from .. fundamentals.conv import conv_3d_transpose

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import batch_normalization


def encoder_1dcovnv_5_points(in_signal):
    layer = conv_1d(in_signal, nb_filter=64, filter_size=5, strides=2, name='conv-fc1')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1, name='conv-fc2')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1, name='conv-fc3')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = tf.reduce_max(layer, axis=1)
    return layer


def decoder_only_with_fc(latent_signal):
    layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = fully_connected(layer, 1024 * 3, activation='linear', weights_init='xavier')
    layer = tf.reshape(layer, [-1, 1024, 3])
    return layer


def decoder_fc_and_1ddeconv(latent_signal):
    layer = fully_connected(latent_signal, 1024, activation='relu', weights_init='xavier')
    layer = fully_connected(layer, 1024, activation='relu', weights_init='xavier')

    layer = tf.tile(layer, [1, 1024])
    layer = tf.reshape(layer, [-1, 1024, 1024])

    layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = conv_1d(layer, nb_filter=64, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = conv_1d(layer, nb_filter=3, filter_size=1, strides=1)
    return layer

# TODO: Panos -> activate them once you fix the input.
#     def _encoder_network(self):
#         # Generate encoder (recognition network), which maps inputs onto a latent space.
#         c = self.configuration
#         layer_sizes = c.encoder_sizes
#         non_linearity = c.transfer_fct
#         layer = non_linearity(fully_connected_layer(self.x, layer_sizes[0], name='encoder_fc_0'))
#         layer = slim.batch_norm(layer)
# 
#         for i in xrange(1, len(c.encoder_sizes)):
#             layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], name='encoder_fc_' + str(i)))
#             layer = slim.batch_norm(layer)
# 
#         return layer
# 
#     def _decoder_network(self):
#         # Generate a decoder which maps points from the latent space back onto the data space.
#         c = self.configuration
#         layer_sizes = c.decoder_sizes
#         non_linearity = c.transfer_fct
#         i = 0
#         layer = non_linearity(fully_connected_layer(self.z, layer_sizes[i], name='decoder_fc_0'))
#         layer = slim.batch_norm(layer)
#         for i in xrange(1, len(layer_sizes) - 1):
#             layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], name='decoder_fc_' + str(i)))
#             layer = slim.batch_norm(layer,)
#         layer = fully_connected_layer(layer, layer_sizes[i + 1], name='decoder_fc_' + str(i + 1))  # Last decoding layer doesn't have a non-linearity.
# 
#         return layer

