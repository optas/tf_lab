import tensorflow as tf
import numpy as np

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, highway_conv_1d 
from tflearn.layers.normalization import batch_normalization

from tf_lab.fundamentals.utils import expand_scope_by_name, replicate_parameter_for_all_layers

def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, b_norm=True,
                            non_linearity=tf.nn.relu, conv_op=conv_1d, regularizer=None,
                            weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):    
    print 'Building Decoder'
    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)
    
    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)
        
        if verbose:
            print name
            print 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list())
        
        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None and i < n_layers - 1:
            layer = non_linearity(layer)
     
        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])
        
        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])
        
        if verbose:
            print layer, np.prod(layer.get_shape().as_list()[1:])
                   
    return layer