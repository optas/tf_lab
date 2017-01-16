import tensorflow as tf
import tensorflow.contrib.slim as slim
from global_variables import *


def autoendoder(in_signal):
    layer = slim.fully_connected(in_signal,nframe*imagelen*imagelen, activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = tf.reshape(layer,[None,nframe,imagelen,imagelen])
    layer = slim.conv2d(layer,24,[4,4],stride=2)
    layer = tf.nn.relu(layer)
    layer = slim.batch_norm(layer,)
    layer = slim.fully_connected(layer, hidden_layer_sizes[1],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[2],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[3],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[4],activation_fn=None)
    layer = tf.tanh(layer)
    return layer


def loss(pred,groundtruth):
    loss = tf.reduce_mean(tf.pow(pred-groundtruth,2))
    tf.summary.scalar("loss",loss)
    return loss
