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


def decoder_only_with_fc(latent_signal):
    layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = fully_connected(layer, 1024 * 3, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.reshape(layer, [-1, 1024, 3])
    return layer


def decoder_fc_and_1ddeconv(latent_signal):
    layer = fully_connected(latent_signal, 1024, activation='relu', weights_init='xavier')
    layer = fully_connected(layer, 1024, activation='relu', weights_init='xavier')

    layer = tf.tile(layer, [1, 1024])
    layer = tf.reshape(layer, [1, 1024, 1024])

    layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = conv_1d(layer, nb_filter=64, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = conv_1d(layer, nb_filter=3, filter_size=1, strides=1)
    return layer
