'''
Created on October 7, 2016

@author: optas
'''

import tensorflow as tf
import numpy as np

def _variable_with_weight_decay(name, shape, stddev, wd=0, dtype=tf.float32, trainable=True):
    '''Creates a Tensor variable initialized with a truncated normal distribution to 
    which optionally weight decay will be applied.

    Args:
        name: name of the variable
        shape: list of ints, shape of the Tensor
        stddev: standard deviation of a truncated Gaussian to be used in initialization
        wd: add L2Loss weight decay multiplied by this float. If 0, weight decay is not added for this Variable.            
    Returns:
        A tensor of the specified dimensions/properties. 
    '''
        
    with tf.device('/cpu:0'):
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        
    if wd > 0:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
    return var


def _bias_variable(shape, init=0.0, trainable=True):
    with tf.device('/cpu:0'):
        return tf.get_variable('bias', shape, initializer=tf.constant_initializer(init), trainable=trainable)


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
    '''Implements a fully connected layer indirectly by using only 2d convolutions.
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
    

def vgg_m_conv(in_signal, keep_prob, stddev = 5e-2):
    '''
    DOI: Return of the Devil in the Details: Delving Deep into Convolutional Nets: Ken Chatfield et al.    
    '''    
    conv = convolutional_layer
    
    layer = conv(in_signal, n_filters=96, filter_size=[7,7], stride=2, padding='SAME', stddev=stddev, name="conv_1")
    layer = max_pool(relu(layer), ksize=(3,3), stride=(2,2), name='max_pool_1')

    layer = conv(layer, n_filters=256, filter_size=[5,5], stride=2, padding='SAME', stddev=stddev, name="conv_2")
    layer = max_pool(relu(layer), ksize=(3,3), stride=(2,2), name='max_pool_2')
                        
    layer = conv(layer, n_filters=512, filter_size=[3,3], stride=1, padding='SAME', stddev=stddev, name="conv_3")
    layer = relu(layer)

    layer = conv(layer, n_filters=512, filter_size=[3,3], stride=1, padding='SAME', stddev=stddev, name="conv_4")
    layer = relu(layer)
    
    layer = conv(layer, n_filters=512, filter_size=[3,3], stride=1, padding='SAME', stddev=stddev, name="conv_5")
    layer = max_pool(relu(layer), ksize=(3,3), stride=(2,2), name='max_pool_3')
    
    layer = fully_connected_layer(layer, 4096, stddev=stddev, name="fc_6")
    layer = dropout(relu(layer), keep_prob)
    
    layer = fully_connected_layer(layer, 4096, stddev=stddev, name="fc_7")
    layer = dropout(relu(layer), keep_prob)

    return layer

def _flat_batch_signal(in_signal):
    '''
    Parameters:
        in_signal (Tensor) of shape (batch_size, dim1, dim2, ...)
    Returns:
        A view of the input signal with shape  (batch_size, prod(dim1, dim2, ...)
        The prod(dim1, dim2, ...) 
    '''
    in_shape = in_signal.get_shape().as_list()   # 1st-dimension is expected to be batch_size
    dim = np.prod(in_shape[1:])
    reshaped = tf.reshape(in_signal, [-1, dim])
    return reshaped, dim 
    