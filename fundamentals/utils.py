'''
Created on February 2, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np


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


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")
