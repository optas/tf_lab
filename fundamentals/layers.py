'''
Created on October 7, 2016

A module containing some commonly used layers of (deep) neural networks.
'''
# TODO: Default initializations are OK, but for convolutions some other initializations will not work.

import tensorflow as tf

from . nn import _flat_batch_signal, _variable_with_weight_decay, _bias_variable
from . initializations import initializer


def max_pool(in_layer, ksize, stride, name):
    ksize = [1, ksize[0], ksize[1], 1]
    strides = [1, stride[0], stride[1], 1]
    return tf.nn.max_pool(in_layer, ksize, strides, padding='SAME', name=name)


def relu(in_layer):
    return tf.nn.relu(in_layer)


def tanh(in_layer):
    return tf.tanh(in_layer)


def safe_log(x, eps=1e-12):
    return tf.log(tf.maximum(x, eps))


def leaky_relu(alpha):
    if not (alpha < 1 and alpha > 0):
        raise ValueError()

    return lambda x: tf.maximum(alpha * x, x)


def dropout(in_layer, keep_prob=0.0):
    if keep_prob == 0.0:
        return in_layer
    else:
        return tf.nn.dropout(in_layer, keep_prob)


def fully_connected_layer(in_layer, out_dim, init={'type': 'glorot'}, wd=0, init_bias=0.0, name='fc'):
    '''Implements a fully connected (fc) layer.
    Args:
        in_layer (tf.Tensor): input signal of the layer
        out_dim (int): output dimension of the fc.
        stddev (float): standard deviation of the gaussian that will be used to initialize the weights.
    '''
    in_layer, fan_in = _flat_batch_signal(in_layer)
    with tf.variable_scope(name):
        shape = [fan_in, out_dim]
        weights = _variable_with_weight_decay('weights', shape, initializer(init, shape), wd=wd)
        biases = _bias_variable([out_dim], init=init_bias)
        out_signal = tf.add(tf.matmul(in_layer, weights), biases, name=name + '_out')
        return out_signal


def fc_with_soft_max_layer(in_layer, out_dim, init={'type': 'glorot'}, wd=0, init_bias=0.0, name='soft_max'):
    in_layer, fan_in = _flat_batch_signal(in_layer)
    with tf.variable_scope(name):
        shape = [fan_in, out_dim]
        weights = _variable_with_weight_decay('weights', shape, initializer(init, shape), wd=wd)
        biases = _bias_variable([out_dim], init=init_bias)
        return tf.nn.softmax(tf.nn.bias_add(tf.matmul(in_layer, weights), biases, name='pre-activation'), name='soft_max')


def conv_2d(in_layer, n_filters, filter_size, stride, padding, name, init={'type': 'glorot_conv2d'}, init_bias=0.0):
    ''' Args:
        n_filters (int): number of convolutional kernels to be used
        filter_size ([int, int]): height and width of each kernel
        stride (int): how many pixels 'down/right' to apply next convolution.
    '''
    with tf.variable_scope(name):
        channels = in_layer.get_shape().as_list()[-1]   # The last dimension of the input layer.
        shape = [filter_size[0], filter_size[1], channels, n_filters]
        kernels = _variable_with_weight_decay('weights', shape, initializer(init, shape))
        biases = _bias_variable([n_filters], init_bias)
        strides = [1, stride, stride, 1]    # same horizontal and vertical strides
        conv = tf.nn.conv2d(in_layer, kernels, strides, padding=padding, name='conv2d')
        bias = tf.nn.bias_add(conv, biases, name='pre-activation')
        out_signal = tf.nn.relu(bias, name=name + '_out')
    return out_signal


def conv_2d_transpose(in_layer, n_filters, filter_size, stride, padding, stddev, name, init_bias=0.0):
    ''' Args:
        n_filters (int): number of convolutional kernels to be used
        filter_size ([int, int]): height and width of each kernel
        stride (int): how many pixels 'down/right' to apply next convolution.
    '''

    with tf.variable_scope(name):
        channels = in_layer.get_shape().as_list()[-1]
        in_shape = tf.shape(in_layer)
        h = ((in_shape[1] - 1) * stride) + 1
        w = ((in_shape[2] - 1) * stride) + 1
        new_shape = [in_shape[0], h, w, n_filters]
        output_shape = tf.pack(new_shape)
        strides = [1, stride, stride, 1]
        shape = [filter_size[0], filter_size[1], n_filters, channels]
        kernels = _variable_with_weight_decay('weights', shape, initializer(shape))
        decon = tf.nn.conv2d_transpose(in_layer, kernels, output_shape, strides=strides, padding=padding, name='conv2d_T')
        biases = _bias_variable([n_filters], init_bias)
        out_signal = tf.nn.bias_add(decon, biases, name=name + '_out')
        return out_signal
