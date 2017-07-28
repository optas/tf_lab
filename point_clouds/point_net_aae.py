'''
Created on July 25, 2017

@author: optas
'''

import time
import numpy as np
import tensorflow as tf
import socket
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected
from tflearn.layers.normalization import batch_normalization

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
                self.old_z = c.encoder(self.x, scope=scope, **c.encoder_args)
                self.bottleneck_size = int(self.old_z.get_shape()[1])
                # To allow for negative values in bneck:
                self.z = fully_connected(self.old_z, self.bottleneck_size, activation='linear', weights_init='xavier', scope=scope)
                self.z = batch_normalization(self.z, scope=scope)

            with tf.variable_scope('decoder') as scope:
                layer = c.decoder(self.z, scope=scope, **c.decoder_args)
                self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.noise, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.z, reuse=True, scope=scope, **disc_kwargs)

            self._create_structural_optimizer()
            self._create_adversarial_optimizer()
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
        c = self.configuration
        self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
        self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))  # encoder will be optimized based on this (+ structural loss).

        train_vars = tf.trainable_variables()
        d_params = [v for v in train_vars if v.name.startswith(self.name + '/discriminator/')]
        g_params = [v for v in train_vars if v.name.startswith(self.name + '/encoder/')]

        self.opt_d = tf.train.AdamOptimizer(c.lr_adv, c.beta_adv).minimize(self.loss_d, var_list=d_params)
        self.opt_g = tf.train.AdamOptimizer(c.lr_adv, c.beta_adv).minimize(self.loss_g, var_list=g_params)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
#         return np.abs(np.random.normal(mu, sigma, (n_samples, ndims)))
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_losses = np.zeros(3)
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, _, _ = train_data.next_batch(batch_size)
            _, loss_s = self.sess.run((self.structural_optimizer, self.structural_loss), feed_dict={self.x: batch_i})

            # Update discriminator.
            noise = self.generator_noise_distribution(batch_size, self.noise_dim, 0, 1)
            feed_dict = {self.x: batch_i, self.noise: noise}
            loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)
            loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)

            # Compute average loss
            epoch_losses += np.array([loss_s, loss_d, loss_g])

        epoch_losses /= n_batches
        duration = time.time() - start_time
        return epoch_losses, duration

    def train(self, train_data, configuration, log_file=None):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.epoch.assign_add(tf.constant(1.0))))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss", loss)
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, 'models.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)
