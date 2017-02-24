'''
Created on February 4, 2017

@author: optas
'''

import tensorflow as tf

from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_1d
from tflearn.layers.normalization import batch_normalization

from . spatial_transformer import transformer as pcloud_spn

try:
    from tflearn.layers.conv import conv_3d_transpose
except:
    print 'Loading manual conv_3d_transpose.'
    from tf_lab.fundamentals.conv import conv_3d_transpose


def encoder_with_convs_and_symmetry(in_signal, layers=[64, 128, 1024], b_norm=True, spn=False, non_linearity=tf.nn.relu, symmetry=tf.reduce_max):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    if spn:
        in_signal = pcloud_spn(in_signal)

    layer = conv_1d(in_signal, nb_filter=layers[0], filter_size=1, strides=1, name='encoder_conv_layer_0')
    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    layer = conv_1d(layer, nb_filter=layers[1], filter_size=1, strides=1, name='encoder_conv_layer_1')
    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    layer = conv_1d(layer, nb_filter=layers[2], filter_size=1, strides=1, name='encoder_conv_layer_2')
    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    layer = symmetry(layer, axis=1)
    return layer


def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu):
    ''' A decoding network which maps points from the latent space back onto the data space.
    '''

    n_layers = len(layer_sizes)
    if n_layers < 2:
        raise ValueError('For single FC decoder use simpler code.')

    layer = fully_connected(latent_signal, layer_sizes[0], activation='linear', weights_init='xavier', name='decoder_fc_0')
    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    for i in xrange(1, n_layers - 1):
        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name='decoder_fc_' + str(i))
        if b_norm:
            layer = batch_normalization(layer)
        layer = non_linearity(layer)

    layer = fully_connected(layer, layer_sizes[n_layers - 1], name='decoder_fc_' + str(n_layers - 1))  # Last decoding layer doesn't have a non-linearity.

    return layer




# def complex_decoder(latent_signal):
#     layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = fully_connected(layer, 32 * 4 * 4 * 4, activation='linear', weights_init='xavier')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     # Virtually treat the signal as 4 x 4 x 4 Voxels, each having 32 channels.
#     layer = tf.reshape(layer, [-1, 4, 4, 4, 32])
# 
#     # Up-sample signal in an 8 x 8 x 8 voxel-space, with 16 channels.
#     layer = conv_3d_transpose(layer, nb_filter=16, filter_size=4, output_shape=[8, 8, 8], strides=2)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     # Up-sample signal in an 16 x 16 x 16 voxel-space, with 8 channels.
#     layer = conv_3d_transpose(layer, nb_filter=8, filter_size=4, output_shape=[16, 16, 16], strides=2)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     # Up-sample signal in an 32 x 32 x 32 voxel-space, with 4 channels.
#     layer = conv_3d_transpose(layer, nb_filter=4, filter_size=4, output_shape=[32, 32, 32], strides=2)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     # Push back signal into a linear 1D vector.
#     layer = tf.reshape(layer, [-1, 32 * 32 * 32, 4])
# 
#     # Convolve every 32 values via 1024 filters.
#     layer = conv_1d(layer, nb_filter=1024, filter_size=32, strides=32)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     layer = conv_1d(layer, nb_filter=64, filter_size=1, strides=1)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     layer = conv_1d(layer, nb_filter=3, filter_size=1, strides=1)
#     return layer



# def encoder_1dcovnv_5_points(in_signal, spn=False):
#     if spn:
#         in_signal = pcloud_spn(in_signal)
#     layer = conv_1d(in_signal, nb_filter=64, filter_size=5, strides=2, name='conv-fc1')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1, name='conv-fc2')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1, name='conv-fc3')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = tf.reduce_max(layer, axis=1)
#     return layer
# 
# 
# def encoder_1dcovnv_1_points_sum(in_signal, spn=False):
#     if spn:
#         in_signal = pcloud_spn(in_signal)
#     layer = conv_1d(in_signal, nb_filter=64, filter_size=1, strides=1)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1)
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = tf.reduce_sum(layer, axis=1)
#     return layer
# 
# 
# def encoder_1dcovnv_5_points_sum(in_signal, spn=False):     # TODO Unify with rest.
#     if spn:
#         in_signal = pcloud_spn(in_signal)
#     layer = conv_1d(in_signal, nb_filter=64, filter_size=5, strides=2, name='conv-fc1')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1, name='conv-fc2')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1, name='conv-fc3')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = tf.reduce_sum(layer, axis=1)
#     return layer
# 
# 
# def decoder_only_with_fc(latent_signal):
#     layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
#     layer = fully_connected(layer, 1024 * 3, activation='linear', weights_init='xavier')
#     layer = tf.reshape(layer, [-1, 1024, 3])
#     return layer
# 
# 
# def decoder_with_three_fc(latent_signal):
#     layer = fully_connected(latent_signal, 512, activation='linear', weights_init='xavier')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
#     layer = batch_normalization(layer)
#     layer = tf.nn.relu(layer)
# 
#     layer = fully_connected(layer, 1024 * 3, activation='linear', weights_init='xavier')
#     layer = tf.reshape(layer, [-1, 1024, 3])
#     return layer