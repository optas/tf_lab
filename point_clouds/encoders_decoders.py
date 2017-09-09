'''
Created on February 4, 2017

@author: optas

'''

import tensorflow as tf

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d
from tflearn.layers.normalization import batch_normalization

from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.utils import expand_scope_by_name, replicate_parameter_for_all_layers


def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                    b_norm=True, spn=False, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                    symmetry=tf.reduce_max, dropout_prob=None, scope=None, reuse=False):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

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

    if dropout_prob is not None and dropout_prob[0] > 0:
        layer = dropout(layer, 1.0 - dropout_prob[0])

    for i in xrange(1, n_layers):
        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i)

        layer = non_linearity(layer)

        if b_norm:
            name += '_bnorm'
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
    return layer


def encoder_with_convs_and_symmetry_and_fc(in_signal, fc_nout, kw_args):
    layer = encoder_with_convs_and_symmetry(in_signal, **kw_args)
    layer = fully_connected(layer, fc_nout, activation='relu', weights_init='xavier')
    return layer


def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, epsilon=0.001, decay=0.99, name=name, reuse=reuse, scope=scope_i)

        layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

    # Last decoding layer doesn't have a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

    return layer
