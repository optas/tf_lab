import tensorflow as tf
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_1d

try:
    from tflearn.layers.conv import conv_3d_transpose
except:
    from .. fundamentals.conv import conv_3d_transpose


def encoder(in_signal):
    layer = conv_1d(in_signal, nb_filter=64, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = tf.reduce_max(layer, axis=1)
    return layer


def decoder(latent_signal):
    layer = fully_connected(latent_signal, 1024, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)
    layer = fully_connected(layer, 32 * 4 * 4 * 4, activation='linear', weights_init='xavier')
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    # Virtually treat the signal as 4 x 4 x 4 Voxels, each having 32 channels.
    layer = tf.reshape(layer, [-1, 4, 4, 4, 32])

    # Up-sample signal in an 8 x 8 x 8 voxel-space, with 16 channels.
    layer = conv_3d_transpose(layer, nb_filter=16, filter_size=4, output_shape=[8, 8, 8], strides=2)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    # Up-sample signal in an 16 x 16 x 16 voxel-space, with 8 channels.
    layer = conv_3d_transpose(layer, nb_filter=8, filter_size=4, output_shape=[16, 16, 16], strides=2)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    # Up-sample signal in an 32 x 32 x 32 voxel-space, with 4 channels.
    layer = conv_3d_transpose(layer, nb_filter=4, filter_size=4, output_shape=[32, 32, 32], strides=2)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    # Push back signal into a linear 1D vector.
    layer = tf.reshape(layer, [-1, 32 * 32 * 32, 4])

    # Convolve every 32 values via 1024 filters.
    layer = conv_1d(layer, nb_filter=1024, filter_size=32, strides=32)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = conv_1d(layer, nb_filter=64, filter_size=1, strides=1)
    layer = batch_normalization(layer)
    layer = tf.nn.relu(layer)

    layer = conv_1d(layer, nb_filter=3, filter_size=1, strides=1)
    return layer


def autoencoder(in_signal):
    latent = encoder(in_signal)
    return decoder(latent)
