'''
Created on Jan 9, 2017

@author: optas
'''

import tensorflow as tf


def _activation_summary(x):
    '''Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: (tf.Tensor)
    Returns:
        nothing
    '''

# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
# session. This helps the clarity of presentation on tensorboard.
# tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tensor_name = x.op.name
    tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
    tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def trainable_variables(in_graph=None):
    if in_graph is None:
        return tf.trainable_variables()
    else:
        return in_graph.get_collection('trainable_variables')


def count_trainable_parameters(in_graph=None):
    if in_graph is None:
        in_graph = tf.get_default_graph()

    total_parameters = 0
    # for variable in tf.trainable_variables():
    for variable in in_graph.get_collection('trainable_variables'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters


def hist_summary_of_trainable(in_graph=None):
    summaries = []
    with tf.device('/cpu:0'):
        for var in trainable_variables(in_graph):
#             summaries.append(tf.histogram_summary(var.op.name, var))
            summaries.append(tf.summary.histogram(var.op.name, var))
    return summaries


def sparsity_summary_of_trainable():
    summaries = []
    with tf.device('/cpu:0'):
        for var in tf.trainable_variables():
            summaries.append(tf.scalar_summary('sparsity_' + var.op.name, tf.nn.zero_fraction(var)))
    return summaries
