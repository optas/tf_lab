import tensorflow as tf
import tensorflow.contrib.slim as slim


hidden = 4
hidden_layer_sizes = [Npoint*4, Npoint, int(0.2*Npoint), Npoint, Npoint*3]


def autoendoder(in_signal):
    in_signal = tf.reshape(in_signal, [-1, Npoint*3])
    layer = slim.fully_connected(in_signal, hidden_layer_sizes[0], activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[1],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[2],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[3],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[4],activation_fn=None)
    layer = tf.tanh(layer)
    return tf.reshape(layer, [-1, Npoint, 3])
