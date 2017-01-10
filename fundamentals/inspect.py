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


def hist_summary_of_trainable():
    summaries = []
    with tf.device('/cpu:0'):
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))
    return summaries


def sparsity_summary_of_trainable():
    summaries = []
    with tf.device('/cpu:0'):
        for var in tf.trainable_variables():
            summaries.append(tf.scalar_summary('sparsity_' + var.op.name, tf.nn.zero_fraction(var)))
    return summaries
