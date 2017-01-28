import tensorflow as tf
import tensorflow.contrib.slim as slim
from global_variables import *

hidden = 4
hidden_layer_sizes = [Npoint, int(Npoint*0.1), int(0.01*Npoint), int(0.1*Npoint), Npoint*3]


def autoencoder(in_signal):
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
    return layer #tf.reshape(layer, [-1, Npoint, 3])


def loss(pred,groundtruth):
    loss = tf.reduce_mean(tf.pow(pred-groundtruth,2))
    tf.summary.scalar("loss",loss)
    return loss
