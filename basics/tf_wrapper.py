'''
Created on October 7, 2016

@author: optas
'''

import tensorflow as tf

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


def max_pool(in_signal, kernel, stride, name):
    ksize = [1, kernel[0], kernel[1], 1]
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
    in_shape = in_signal.get_shape().as_list()
    dim = 1
    for d in in_shape[1:]:
        dim *= d
    reshape = tf.reshape(in_signal, [-1, dim])
    #TODO Compare with this, which is default of Tutorial,. reshape = tf.reshape(in_signal, [BATCH_SIZE, -1])
    #dim = reshape.get_shape()[1].value
    with tf.variable_scope(name) as scope:    
        weights = _variable_with_weight_decay('weights', [dim, out_dim], stddev=stddev, wd=wd) 
        biases = _bias_variable([out_dim], init=init_bias)
        out_signal = tf.add(tf.matmul(reshape, weights), biases, name=scope.name+'_out')
        return out_signal


def convolutional_layer(in_signal, filters, field_size, stride, padding, stddev, name, bias_init=0.0):
    with tf.variable_scope(name) as scope:
        channels = in_signal.get_shape().as_list()[-1]        
        kernels = _variable_with_weight_decay('weights', shape=[field_size[0], field_size[1], channels, filters], stddev=stddev)
        biases = _bias_variable([filters], bias_init)        
        conv = tf.nn.conv2d(in_signal, kernels, [1, stride, stride, 1], padding=padding, name='conv2d')                
        bias = tf.nn.bias_add(conv, biases, name='activation_in')        
        out_signal = tf.nn.relu(bias, name=scope.name + '_out')
    return out_signal
        
    
def vgg_m_conv(in_signal, keep_prob):
    '''
    DOI: Return of the Devil in the Details: Delving Deep into Convolutional Nets: Ken Chatfield et al.    
    '''    
    conv = convolutional_layer
    
    layer = conv(in_signal, filters=96, field_size=7, stride=2, padding='SAME', name="conv_1")
    layer = max_pool(relu(layer), kernel=(3,3), stride=(2,2))

    layer = conv(layer, filters=256, field_size=5, stride=2, padding='SAME', name="conv_2")
    layer = max_pool(relu(layer), kernel=(3,3), stride=(2,2))
                        
    layer = conv(layer, filters=512, field_size=3, stride=1, padding='SAME', name="conv_3")
    layer = relu(layer)

    layer = conv(layer, filters=512, field_size=3, stride=1, padding='SAME', name="conv_4")
    layer = relu(layer)
    
    layer = conv(layer, filters=512, field_size=3, stride=1, padding='SAME', name="conv_5")
    layer = max_pool(relu(layer), kernel=(3,3), stride=(2,2))
    
    layer = fully_connected_layer(layer, 4096, name="fc_6")
    layer = dropout(relu(layer), keep_prob)
    
    layer = fully_connected_layer(layer, 4096, name="fc_7")
    layer = dropout(relu(layer), keep_prob)

    return layer