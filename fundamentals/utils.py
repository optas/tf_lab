'''
Created on February 2, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np
from six import string_types, integer_types


def expand_scope_by_name(scope, name):
    """ expand_scope_by_name.

    tflearn seems to not append the name in the scope automatically.
    """

    if isinstance(scope, string_types):
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
    if parameter is not None and len(parameter) != n_layers:
        if len(parameter) != 1:
            raise ValueError()
        parameter = np.array(parameter)
        parameter = parameter.repeat(n_layers).tolist()
    return parameter


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, np.ndarray, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def count_cmp_to_value(in_tensor, bound_val, comparator=tf.equal, axis=None):
    ''' count number of elements of tensors that are bigger/smaller etc. than a `bound_val`.
    '''
    elements_equal_to_value = comparator(in_tensor, bound_val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints, axis=axis)
    count = tf.cast(count, tf.float32)
    return count


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False):
    '''Even at zero it will return epsilon.
    Reminder: l2_norm has no derivative at 0.0.
    '''
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank. If list is provided, one of the ranks listed is expected to be correct.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            'For the tensor `%s` in scope `%s`, the actual rank '
            '`%d` (shape = %s) is not equal to the expected rank `%s`' %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

        
def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
          specified and the `tensor` has a different rank, and exception will be
          thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape