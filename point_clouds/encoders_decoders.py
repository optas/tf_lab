'''
Created on February 4, 2017

@author: optas

TODO - Clean up new/old versions.
'''

import tensorflow as tf

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d
from tflearn.layers.normalization import batch_normalization

from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.utils import expand_scope_by_name

try:
    from tflearn.layers.conv import conv_3d_transpose
except:
    print('conv_3d_transpose function not found.')


def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1, 1, 1, 1], strides=[1, 1, 1, 1],
                                    b_norm=True, spn=False, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                    symmetry=tf.reduce_max, dropout_prob=None, scope=None, reuse=False):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    n_layers = len(n_filters)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    if spn:
        transformer = pcloud_spn(in_signal)
        in_signal = tf.batch_matmul(in_signal, transformer)
        print 'Spatial transformer was activated.'

    name = 'encoder_conv_layer_0'
    scope_i = expand_scope_by_name(scope, name)
    layer = conv_1d(in_signal, nb_filter=n_filters[0], filter_size=filter_sizes[0], strides=strides[0], regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i)

    if b_norm:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

    layer = non_linearity(layer)

    for i in xrange(1, n_layers):
        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

        layer = non_linearity(layer)

    if dropout_prob is not None:
        layer = dropout(layer, dropout_prob)

    layer = symmetry(layer, axis=1)
    return layer


def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu, reuse=False, scope=None):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''

    n_layers = len(layer_sizes)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    name = 'decoder_fc_0'
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(latent_signal, layer_sizes[0], activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)

    if b_norm:
        name = 'decoder_fc_0_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

    layer = non_linearity(layer)

    for i in xrange(1, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

        layer = non_linearity(layer)

    # Last decoding layer doesn't have a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_i)

    return layer





# def encoder_with_convs_and_symmetry(in_signal, layer_sizes=[64, 128, 1024], b_norm=True, spn=False, non_linearity=tf.nn.relu, symmetry=tf.reduce_max, dropout_prob=None):
#     '''An Encoder (recognition network), which maps inputs onto a latent space.
#     '''
#     if spn:
#         transformer = pcloud_spn(in_signal)
#         in_signal = tf.batch_matmul(in_signal, transformer)
#         print 'Spatial transformer was activated.'
# 
#     layer = conv_1d(in_signal, nb_filter=layer_sizes[0], filter_size=1, strides=1, name='encoder_conv_layer_0')
# 
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     layer = conv_1d(layer, nb_filter=layer_sizes[1], filter_size=1, strides=1, name='encoder_conv_layer_1')
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     layer = conv_1d(layer, nb_filter=layer_sizes[2], filter_size=1, strides=1, name='encoder_conv_layer_2')
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     if dropout_prob is not None:
#         layer = dropout(layer, dropout_prob)
# 
#     layer = symmetry(layer, axis=1)
#     return layer


# def decoder_in_voxel_space_v0(latent_signal, b_norm=True, non_linearity=tf.nn.relu):
#     # TODO make it work with variable input/filter sizes
#     layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     layer = fully_connected(layer, 32 * 4 * 4 * 4, activation='linear', weights_init='xavier')
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     # Virtually treat the signal as 4 x 4 x 4 Voxels, each having 32 channels.
#     layer = tf.reshape(layer, [-1, 4, 4, 4, 32])
# 
#     # Up-sample signal in an 8 x 8 x 8 voxel-space, with 16 channels.
#     layer = conv_3d_transpose(layer, nb_filter=16, filter_size=4, output_shape=[8, 8, 8], strides=2)
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     # Up-sample signal in an 16 x 16 x 16 voxel-space, with 8 channels.
#     layer = conv_3d_transpose(layer, nb_filter=8, filter_size=4, output_shape=[16, 16, 16], strides=2)
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     # Up-sample signal in an 32 x 32 x 32 voxel-space, with 1 channel.
#     layer = conv_3d_transpose(layer, nb_filter=1, filter_size=4, output_shape=[32, 32, 32], strides=2)
#     return layer


# TODO The code below will be cleaned-up, deleted ASAP.

# def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu, reuse=False):
#     ''' A decoding network which maps points from the latent space back onto the data space.
#     '''
# 
#     n_layers = len(layer_sizes)
#     if n_layers < 2:
#         raise ValueError('For an FC decoder with single a layer use simpler code.')
# 
#     layer = fully_connected(latent_signal, layer_sizes[0], activation='linear', weights_init='xavier', name='decoder_fc_0', reuse=reuse)
#     if b_norm:
#         layer = batch_normalization(layer)
#     layer = non_linearity(layer)
# 
#     for i in xrange(1, n_layers - 1):
#         layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name='decoder_fc_' + str(i), reuse=reuse)
#         if b_norm:
#             layer = batch_normalization(layer)
#         layer = non_linearity(layer)
# 
#     # Last decoding layer doesn't have a non-linearity.
#     layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name='decoder_fc_' + str(n_layers - 1), reuse=reuse)
# 
#     return layer
