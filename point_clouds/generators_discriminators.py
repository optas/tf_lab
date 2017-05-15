'''
Created on May 11, 2017

@author: optas
'''

import numpy as np
import tensorflow as tf
from tflearn.layers.core import fully_connected

from . encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only
from .. fundamentals.utils import expand_scope_by_name, leaky_relu


def mlp_discriminator(in_signal, non_linearity=tf.nn.relu, reuse=False, scope=None):
    encoder_args = {'n_filters': [64, 128, 256, 256, 512], 'filter_sizes': [1, 1, 1, 1, 1], 'strides': [1, 1, 1, 1, 1]}
    encoder_args['reuse'] = reuse
    encoder_args['scope'] = scope
    encoder_args['non_linearity'] = non_linearity
    layer = encoder_with_convs_and_symmetry(in_signal, **encoder_args)

    name = 'decoding_logits'
    scope_e = expand_scope_by_name(scope, name)
    d_logit = decoder_with_fc_only(layer, layer_sizes=[128, 64, 1], reuse=reuse, scope=scope_e)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def convolutional_discriminator(in_signal, non_linearity=tf.nn.relu, reuse=False, scope=None):
    encoder_args = {'n_filters': [128, 128, 256, 512], 'filter_sizes': [40, 20, 10, 10], 'strides': [1, 2, 2, 1]}
    encoder_args['reuse'] = reuse
    encoder_args['scope'] = scope
    encoder_args['non_linearity'] = non_linearity
    layer = encoder_with_convs_and_symmetry(in_signal, **encoder_args)

    name = 'decoding_logits'
    scope_e = expand_scope_by_name(scope, name)
    d_logit = decoder_with_fc_only(layer, layer_sizes=[128, 64, 1], reuse=reuse, scope=scope_e)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def point_cloud_generator(z, n_points, layer_sizes=[64, 128, 512, 1024], b_norm=True):
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, b_norm=b_norm)
    out_signal = tf.nn.relu(out_signal)
    out_signal = fully_connected(out_signal, np.prod([n_points, 3]), activation='linear', weights_init='xavier')
    out_signal = tf.reshape(out_signal, [-1, n_points, 3])
    return out_signal


def latent_code_generator(z, out_dim, layer_sizes=[64, 128], b_norm=True):
    layer_sizes = layer_sizes + out_dim
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, b_norm=b_norm)
    out_signal = tf.nn.relu(out_signal)
    return out_signal


def latent_code_discriminator(in_singnal, layer_sizes=[64, 128, 256, 256, 512], b_norm=True, non_linearity=tf.nn.relu, reuse=False, scope=None):
    layer_sizes = layer_sizes + [1]
    d_logit = decoder_with_fc_only(in_singnal, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm, reuse=reuse, scope=scope)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit