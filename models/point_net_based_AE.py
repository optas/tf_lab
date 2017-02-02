import tensorflow as tf
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_1d, conv_2d, conv_3d, conv_3d_transpose


def encoder(in_signal):
    layer = in_signal
    layer = conv_1d(layer, 64, 1, 1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, 128, 1, 1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, 1024, 1, 1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = tf.reduce_max(layer, 1)
    return layer


def decoder(latent_signal):
    layer = fully_connected(latent_signal, 1024, activation='relu', weights_init='xavier')
    layer = fully_connected(layer, 32 * 4 * 4 * 4, activation='relu', weights_init='xavier')
    layer = tf.reshape(layer, [-1, 4, 4, 4, 32])
    layer = conv_3d_transpose(layer, 16, 4, [8, 8, 8], strides=2, activation='relu')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_3d_transpose(layer, 8, 4, [16, 16, 16], strides=2, activation='relu')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_3d_transpose(layer, 4, 4, [32, 32, 32], strides=2, activation='relu')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = tf.reshape(layer, [-1, 32 * 32 * 32, 4])
    layer = conv_1d(layer, 1024, 32, strides=32)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, 512, 1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, 218, 1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, 128, 1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, 32, 1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, 3, 1)
    return layer


def autoencoder(in_signal):
    latent = encoder(in_signal)
    return decoder(latent)
