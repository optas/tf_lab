'''
Created on February 2, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np
from sympy.integrals.risch import NonElementaryIntegral


def expand_scope_by_name(scope, name):
    """ expand_scope_by_name.

    tflearn seems to not append the name in the scope automatically.
    """

    if isinstance(scope, basestring):
        scope += '/' + name
        return scope

    if scope is not None:
        return scope.name + '/' + name
    else:
        return scope


def format_scope_name(scope_name, prefix, suffix):
    """ format_scope_name.

    Add a prefix and a suffix to a scope name.
    """

    if prefix is not "":
        if not prefix[-1] == "/":
            prefix += "/"
    if suffix is not "":
        if not suffix[0] == "/":
            suffix = "/" + suffix
    return prefix + scope_name + suffix


def replicate_parameter_for_all_layers(parameter, n_layers):
    if parameter is not None and len(parameter) is not n_layers:
        parameter = np.array(parameter)
        parameter = parameter.repeat(n_layers).tolist()
    return parameter


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def count_cmp_to_value(in_tensor, bound_val, comparator=tf.equal):
    ''' count number of elements of tensors that are bigger/smaller etc. than a `bound_val`.
    '''
    elements_equal_to_value = comparator(in_tensor, bound_val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    count = tf.cast(count, tf.float32)
    return count
