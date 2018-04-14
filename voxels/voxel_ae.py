'''
Created on Dec 14, 2017

@author: optas
'''

import time
import tensorflow as tf

from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_3d, conv_3d_transpose
from tflearn.layers.normalization import batch_normalization

from .. point_clouds.autoencoder import AutoEncoder


class Voxel_AE(AutoEncoder):
    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c
        AutoEncoder.__init__(self, name, graph, configuration)
        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.x_reconstr = c.decoder(self.z, **c.decoder_args)
            self._create_loss()
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name),
                                        max_to_keep=c.saver_max_to_keep)

        self.start_session()

    def _create_loss(self):
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.x_reconstr)
        self.loss = tf.reduce_mean(self.loss)
        self._setup_optimizer()

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, batch_size, only_fw=False):
        ''' train_data: first returned argument of next_batch must be voxel grid.
        '''
        n_examples = train_data.n_examples
        epoch_loss = 0.

        if type(batch_size) is not int:   # TODO temp fix.
            batch_size = batch_size.batch_size

        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i = train_data.next_batch(batch_size)[0]
            _, loss = fit(batch_i)
            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration


def iclr_conv_encoder(in_signal, resolution, b_neck=64):
    ''' Used in rebuttal of ICLR.
    '''
    layer = in_signal
    layer = conv_3d(layer, nb_filter=32, filter_size=6, strides=2, activation='relu')
    layer = conv_3d(layer, nb_filter=32, filter_size=6, strides=2, activation='relu')
    layer = batch_normalization(layer)
    layer = conv_3d(layer, nb_filter=64, filter_size=4, strides=2, activation='relu')
    if resolution == 32:
        layer = conv_3d(layer, nb_filter=64, filter_size=2, strides=2, activation='relu')
        layer = batch_normalization(layer)
    elif resolution == 64:
        layer = conv_3d(layer, nb_filter=64, filter_size=4, strides=2, activation='relu')
        layer = batch_normalization(layer)
        layer = conv_3d(layer, nb_filter=64, filter_size=2, strides=2, activation='relu')
    else:
        raise ValueError()
    layer = conv_3d(layer, nb_filter=b_neck, filter_size=2, strides=2, activation='relu')
    return layer


def iclr_conv_decoder(in_signal, resolution):
    ''' Used in rebuttal of ICLR.
    '''
    layer = in_signal
    layer = conv_3d_transpose(layer, nb_filter=64, filter_size=2, strides=2, output_shape=[2, 2, 2], activation='relu')
    layer = conv_3d_transpose(layer, nb_filter=32, filter_size=4, strides=2, output_shape=[4, 4, 4], activation='relu')
    layer = batch_normalization(layer)
    layer = conv_3d_transpose(layer, nb_filter=32, filter_size=6, strides=2, output_shape=[8, 8, 8], activation='relu')
    if resolution == 32:
        layer = conv_3d_transpose(layer, nb_filter=1, filter_size=8, strides=4, output_shape=[32, 32, 32], activation='linear')
    elif resolution == 64:
        layer = conv_3d_transpose(layer, nb_filter=32, filter_size=6, strides=2, output_shape=[16, 16, 16], activation='relu')
        layer = batch_normalization(layer)
        layer = conv_3d_transpose(layer, nb_filter=32, filter_size=8, strides=2, output_shape=[32, 32, 32], activation='relu')
        layer = conv_3d_transpose(layer, nb_filter=1, filter_size=8, strides=2, output_shape=[64, 64, 64], activation='linear')
    else:
        raise ValueError
    return layer
