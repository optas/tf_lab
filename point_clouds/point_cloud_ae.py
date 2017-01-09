import tensorflow as tf
import numpy as np

from .. fundamentals.layers import fully_connected_layer, relu, tanh


class Configuration(object):
    n_points = 3000
    original_embedding = 3  # Points leave in 3D space.

    hidden_out_sizes = [n_points,
                        int(np.around(0.5 * n_points)),
                        int(np.around(0.1 * n_points)),
                        int(np.around(0.5 * n_points)),
                        int(np.around(n_points * original_embedding))
                        ]

    batch_size = 4
    training_epochs = 1000


def in_out_placeholders(configuration):
    n = configuration.n_points
    e = configuration.original_embedding
    b = configuration.batch_size
    in_signal = tf.placeholder(dtype=tf.float32, shape=(b, n, e), name='input_pclouds')
    gt_signal = tf.placeholder(dtype=tf.float32, shape=(b, n, e), name='output_pclouds')
    return in_signal, gt_signal


def autoendoder_with_fcs_only(in_signal, configuration):
    hs = configuration.hidden_out_sizes
    n = configuration.n_points
    e = configuration.original_embedding

    in_signal = tf.reshape(in_signal, [-1, n * e])
    layer = fully_connected_layer(in_signal, hs[0], stddev=0.01, name='fc_1')
    layer = fully_connected_layer(relu(layer), hs[1], stddev=0.01, name='fc_2')
    layer = fully_connected_layer(relu(layer), hs[2], stddev=0.01, name='fc_3')
    layer = fully_connected_layer(relu(layer), hs[3], stddev=0.01, name='fc_4')
    layer = fully_connected_layer(layer, hs[4], stddev=0.01, name='fc_5')
    layer = tanh(layer)
    return tf.reshape(layer, [-1, n, e])
