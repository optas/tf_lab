'''
Created on July 25, 2017

@author: optas
'''

import time
import numpy as np
import tensorflow as tf
import socket

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from general_tools.in_out.basics import create_dir

from . autoencoder import AutoEncoder
from . in_out import apply_augmentations
from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.loss import Loss
from .. fundamentals.inspect import count_trainable_parameters

try:
    if socket.gethostname() == socket.gethostname() == 'oriong2.stanford.edu':
        from .. external.oriong2.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.oriong2.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
    else:
        from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded.')


class PointNetAdversarialAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder replicating the architecture of Charles and Hao paper.
    '''

    def __init__(self, name, configuration, noise_dim, discriminator, disc_kwargs={}, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()     # TODO change to make a new graph.

        c = configuration
        self.configuration = c
        self.noise_dim = noise_dim
        self.discriminator = discriminator

        AutoEncoder.__init__(self, name, configuration)

        with tf.variable_scope(name):
            self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])     # Noise vector.

            with tf.variable_scope('encoder') as scope:
                self.z = c.encoder(self.x, scope=scope, **c.encoder_args)

            with tf.variable_scope('decoder') as scope:
                layer = c.decoder(self.z, scope=scope, **c.decoder_args)
                self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.z, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.noise, reuse=True, scope=scope, **disc_kwargs)

            self._create_structural_optimizer()
            self._create_adversarial_optimizer()

            self.bottleneck_size = int(self.z.get_shape()[1])
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):  # TODO - mitigate hasaatr
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_structural_optimizer(self):
        c = self.configuration
        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.structural_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.structural_loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        self.structural_optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.structural_loss)

    def _create_adversarial_optimizer(self):
        self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
        self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))  # encoder will be optimized based on this (+ structural loss).

        train_vars = tf.trainable_variables()
        d_params = [v for v in train_vars if v.name.startswith(self.name + '/discriminator/')]
        g_params = [v for v in train_vars if v.name.startswith(self.name + '/encoder/')]
        print g_params
        self.opt_d = tf.train.AdamOptimizer(0.0001, 0.9).minimize(self.loss_d, var_list=d_params)
        self.opt_g = tf.train.AdamOptimizer(0.0001, 0.9).minimize(self.loss_g, var_list=g_params)

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if self.is_denoising:
                loss, _ = self.partial_fit(batch_i, original_data)
            else:
                loss, _ = self.partial_fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration
