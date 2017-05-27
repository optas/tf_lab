'''
Created on February 2, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np


def expand_scope_by_name(scope, name):
    '''
    tflearn seems to not append the name in the scope automatically.
    '''

    if isinstance(scope, basestring):
        scope += '/' + name
        return scope

    if scope is not None:
        return scope.name + '/' + name
    else:
        return scope


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def leaky_relu(alpha):
    if not (alpha < 1 and alpha > 0):
        raise ValueError()

    return lambda x: tf.maximum(alpha * x, x)

