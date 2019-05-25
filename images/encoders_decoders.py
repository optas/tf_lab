'''
Created on Feb 12, 2018

@author: optas
'''
import numpy as np
import tensorflow as tf

from tflearn.layers.conv import conv_2d, max_pool_2d, conv_2d_transpose
from tflearn.layers.normalization import batch_normalization

from .. point_clouds.encoders_decoders import decoder_with_fc_only
from .. fundamentals.utils import expand_scope_by_name, replicate_parameter_for_all_layers



def conv_encoder_params(version):
    if version == 1:
        # Preliminary for geometric-images.
        filter_sizes = [[7, 7], [5, 5], [3, 3], [3, 3]]
        n_filters = [96, 128, 128, 128]
        strides = [2, 2, 1, 1]
        pool_kernels = [[3, 3], [3, 3], None, None]
        pool_strides = [2, 2, 2, 2]
    return filter_sizes, n_filters, strides, pool_kernels, pool_strides


def conv_based_encoder(in_signal, n_filters, filter_sizes, strides=[1], b_norm=[True], 
                       non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                       pool=max_pool_2d, pool_kernels=None, pool_strides=None, scope=None,
                       reuse=False, padding='same', verbose=False, conv_op=conv_2d):
    
    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    b_norm = replicate_parameter_for_all_layers(b_norm, n_layers)
    
    container = []

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], 
                        regularizer=regularizer, weight_decay=weight_decay, 
                        name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),


        if b_norm[i]:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))
                        

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_kernels is not None:
            if pool_kernels[i] is not None:
                layer = pool(layer, kernel_size=pool_kernels[i], strides=pool_strides[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

        container.append(layer)
    return container[-1], container
