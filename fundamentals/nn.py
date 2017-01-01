'''
Created on October 7, 2016

A module containing the lowest level ingredients necessary to build higher level (nn) abstractions like layers.   
'''

import tensorflow as tf
import numpy as np

def _variable_with_weight_decay(name, shape, stddev, wd=0, dtype=tf.float32, trainable=True):
    '''Creates a Tensor variable initialized with a truncated normal distribution to 
    which optionally weight decay will be applied.

    Args:
        name        (string): name of the variable
        shape (list of ints): shape of the Tensor
        stddev       (float): standard deviation of a truncated Gaussian to be used in initialization
        wd:          (float): L2Loss weight decay multiplied by this float. If 0, weight decay is not added for this Variable.            

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

    

def _flat_batch_signal(in_signal):
    '''
    Args:
        in_signal (Tensor):  shape (batch_size, dim1, dim2, ...)
        
    Returns:
        1. A view of the input signal with shape  (batch_size, prod(dim1, dim2, ...)
        2. The prod(dim1, dim2, ...) 
    '''
    in_shape = in_signal.get_shape().as_list()   # 1st-dimension is expected to be batch_size
    dim = np.prod(in_shape[1:])
    reshaped = tf.reshape(in_signal, [-1, dim])
    return reshaped, dim 
    