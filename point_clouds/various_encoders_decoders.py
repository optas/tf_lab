'''
Created on February 4, 2017

@author: optas
'''

import tensorflow as tf

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import batch_normalization

from . spatial_transformer import transformer as pcloud_spn

try:
    from tflearn.layers.conv import conv_3d_transpose
except:
    print 'Loading manual conv_3d_transpose.'
    from tf_lab.fundamentals.conv import conv_3d_transpose


def encoder_1dcovnv_5_points(in_signal, spn=False):
    if spn:
        in_signal = pcloud_spn(in_signal)
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


def encoder_1dcovnv_1_points_sum(in_signal, spn=False):
    if spn:
        in_signal = pcloud_spn(in_signal)
    layer = conv_1d(in_signal, nb_filter=64, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = tf.reduce_sum(layer, axis=1)
    return layer


def encoder_1dcovnv_5_points_sum(in_signal, spn=False):     # TODO Unify with rest.
    if spn:
        in_signal = pcloud_spn(in_signal)
    layer = conv_1d(in_signal, nb_filter=64, filter_size=5, strides=2, name='conv-fc1')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1, name='conv-fc2')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1, name='conv-fc3')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = tf.reduce_sum(layer, axis=1)
    return layer


def decoder_only_with_fc(latent_signal):
    layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = fully_connected(layer, 1024 * 3, activation='linear', weights_init='xavier')
    layer = tf.reshape(layer, [-1, 1024, 3])
    return layer


def decoder_with_three_fc(latent_signal):
    layer = fully_connected(latent_signal, 512, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = fully_connected(layer, 1024 * 3, activation='linear', weights_init='xavier')
    layer = tf.reshape(layer, [-1, 1024, 3])
    return layer


def decoder_with_fc_only(latent_signal, layer_sizes, b_norm=True, non_linearity=tf.nn.relu):

    if len(layer_sizes) < 2:
        raise ValueError('For single FC decoder use simpler code.')

    layer = fully_connected(latent_signal, layer_sizes[0], activation='linear', weights_init='xavier', name='decoder_fc_0')
    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    for i in xrange(1, len(layer_sizes) - 1):
        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name='decoder_fc_' + str(i))
        if b_norm:
            layer = batch_normalization(layer)
        layer = non_linearity(layer)

    layer = fully_connected(layer, layer_sizes[i + 1], name='decoder_fc_' + str(i + 1))  # Last decoding layer doesn't have a non-linearity.

    return layer
