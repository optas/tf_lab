'''
Created on Jan 9, 2018

@author: optas
'''

import tensorflow as tf
from tf_lab.point_clouds.encoders_decoders import encoder_with_convs_and_symmetry_new, decoder_with_fc_only
from tf_lab.fundamentals.inspect import count_trainable_parameters
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d


from .in_out import classes_of_tasks

n_best_pc_parms = 34124


def pc_net(n_pc_points, do_regression, n_classes, n_filters, n_neurons):
    with tf.variable_scope('pc_based_net'):
        feed_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points, 3))
        layer = encoder_with_convs_and_symmetry_new(feed_pl, n_filters=n_filters, b_norm=False)
        n_neurons = n_neurons + [n_classes]
        net_out = decoder_with_fc_only(layer, n_neurons, b_norm=False)
        if do_regression:
            net_out = tf.nn.relu(net_out)
    return net_out, feed_pl


def diff_mlp_net(n_cons, task):
    n_classes = classes_of_tasks(task)
    _, last_nn = start_end_of_nets(task, n_classes)
    f_layer = mlp_neurons_on_first_layer(n_cons)
    with tf.variable_scope('mlp_diff_based_net'):
        feed_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons))
        layer = fully_connected(feed_pl, f_layer, activation='relu', weights_init='xavier')
        layer = fully_connected(layer, 50, activation='relu', weights_init='xavier')
        layer = fully_connected(layer, 100, activation='relu', weights_init='xavier')
        net_out = fully_connected(layer, n_classes, activation=last_nn, weights_init='xavier')
        n_tp = count_trainable_parameters()
        print '#PARAMS ', n_tp
        assert (n_tp <= 0.01 * n_best_pc_parms + n_best_pc_parms)
        assert (n_tp >= n_best_pc_parms - 0.01 * n_best_pc_parms)
    return net_out, feed_pl


def diff_conv_net(n_cons, task):
    n_classes = classes_of_tasks(task)
    _, last_nn = start_end_of_nets(task, n_classes)
    feed_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons))
    layer = tf.expand_dims(feed_pl, -1)
    layer = conv_2d(layer, nb_filter=6, filter_size=4, strides=2, activation='relu')
    layer = conv_2d(layer, nb_filter=7, filter_size=2, strides=1, activation='relu')
    net_out = fully_connected(layer, n_classes, activation=last_nn, weights_init='xavier')
    return net_out, feed_pl


def start_end_of_nets(task, n_classes):
    if task == 'regression':
        labels_pl = tf.placeholder(tf.float32, shape=[None, n_classes])
        last_nn = 'relu'
    else:
        labels_pl = tf.placeholder(tf.int64, shape=[None])
        last_nn = 'linear'
    return labels_pl, last_nn


def pc_versions(ver):
    if ver == 'v1':
        n_filters = [32, 64, 64]
        n_neurons = [64]
    elif ver == 'v2':
        n_filters = [64, 128, 128]
        n_neurons = [64]
    elif ver == 'v3':
        n_filters = [64, 128, 128]
        n_neurons = [64, 128]
    else:
        assert(False)
    return n_filters, n_neurons


def mlp_neurons_on_first_layer(n_cons):
    if n_cons == 5:
        f_layer = 369
    elif n_cons == 10:
        f_layer = 185
    elif n_cons == 20:
        f_layer = 62
    elif n_cons == 30:
        f_layer = 29
    elif n_cons == 40:
        f_layer = 17
    elif n_cons == 50:
        f_layer = 11
    return f_layer