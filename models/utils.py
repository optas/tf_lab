'''
Created on February 3, 2017

@author: optas
'''

import tensorflow as tf


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
