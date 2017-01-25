'''
Created on January 13, 2017

@author: optas
#TODO Fix dispatcher. Expand.
'''

import numpy as np
import tensorflow as tf


def initializer(args):
    return glorot_initializer(*args)


def truncated_normal_initializer(stddev, dtype=tf.float32):
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def glorot_initializer(fan_in, fan_out, constant=1.0, uniform=True, dtype=tf.float32):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    SEE: https://github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
    '''
    with tf.device('/cpu:0'):
        if uniform:
            init_range = constant * np.sqrt(6.0 / (fan_in + fan_out))
            return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)
        else:
            stddev = constant * np.sqrt(2.0 / (fan_in + fan_out))
            return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def orthogonal_initializer(shape, scale=1.1):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    SEE: https://github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]
