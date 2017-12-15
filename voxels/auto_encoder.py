'''
Created on Dec 14, 2017

@author: optas
'''

import time
import tensorflow as tf
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from general_tools.in_out.basics import create_dir

from .. point_clouds.autoencoder import AutoEncoder


class Voxel_Based_AE(AutoEncoder):
    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c
        AutoEncoder.__init__(self, name, graph, configuration)
        self.z = c.encoder(self.x)
        self.x_reconstr = c.decoder(self.z)
        self._create_loss()
        init = tf.global_variables_initializer()    # TODO -> make part of Neural_Net
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
        n_examples = train_data.n_examples
        epoch_loss = 0.
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, _ = train_data.next_batch(batch_size)
            _, loss = fit(batch_i)
            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration
