'''
Created on October 7, 2016

A module containing some commonly user layrs of (deep) neural networks. 
'''

import tensorflow as tf
import numpy as np

from . nn import _flat_batch_signal,  _variable_with_weight_decay, _bias_variable 

def max_pool(in_signal, ksize, stride, name):
    ksize = [1, ksize[0], ksize[1], 1]
    strides = [1, stride[0], stride[1], 1]
    return tf.nn.max_pool(in_signal, ksize, strides, padding='SAME', name=name)


def relu(in_signal) :
    return tf.nn.relu(in_signal)


def dropout(in_signal, keep_prob=0.0):
    if keep_prob == 0.0:
        return in_signal
    else:
        return tf.nn.dropout(in_signal, keep_prob)


def fully_connected_layer(in_signal, out_dim, stddev, wd=0, init_bias=0.0, name='fc'):    
    in_signal, dim = _flat_batch_signal(in_signal)
    with tf.variable_scope(name) as scope:    
        weights = _variable_with_weight_decay('weights', [dim, out_dim], stddev=stddev, wd=wd) 
        biases = _bias_variable([out_dim], init=init_bias)
        out_signal = tf.add(tf.matmul(in_signal, weights), biases, name=scope.name+'_out')
        return out_signal


def convolutional_layer(in_signal, n_filters, filter_size, stride, padding, stddev, name, init_bias=0.0):
    ''' 
    Parameters:    
        n_filters : (int) number of convolutional kernels to be used
        filter_size: [int, int] height and width of each kernel
        stride: (int) how many pixels 'down/right' to apply next convolution.     
    '''
    with tf.variable_scope(name) as scope:
        channels = in_signal.get_shape().as_list()[-1]        
        kernels = _variable_with_weight_decay('weights', shape=[filter_size[0], filter_size[1], channels, n_filters], stddev=stddev)
        biases = _bias_variable([n_filters], init_bias)
        strides = [1, stride, stride, 1] # same horizontal and vertical strides
        conv = tf.nn.conv2d(in_signal, kernels, strides, padding=padding, name='conv2d')                
        bias = tf.nn.bias_add(conv, biases, name='activation_in')        
        out_signal = tf.nn.relu(bias, name=scope.name + '_out')
    return out_signal
        
    
def fully_conected_via_convolutions(in_layer, out_dim, stddev, init_bias, name):
    '''Implements a fully connected layer -indirectly- by using only 2d convolutions.
    '''
    with tf.variable_scope(name) as scope:
        in_shape = in_layer.get_shape().as_list()
        vector_dim = np.prod(in_shape[1:])
        batch_dim = in_shape[0]            
        in_layer = tf.reshape(in_layer, [batch_dim, 1, 1, vector_dim])
        kernel = _variable_with_weight_decay('weights', [1, 1, vector_dim, out_dim], stddev=stddev)
        conv = tf.nn.conv2d(in_layer, kernel, [1, 1, 1, 1], padding='SAME')
        conv = tf.reshape(conv, [batch_dim, out_dim])        
        biases = _bias_variable([out_dim], init_bias)
        out_layer = tf.nn.bias_add(conv, biases, name='activation_in')
    return out_layer    
    