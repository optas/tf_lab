'''
Created on February 1, 2017

@author: optas
'''

import numpy as np
import time
import os.path as osp
import tensorflow as tf
from tflearn.layers.core import fully_connected as fc_layer

from general_tools.in_out.basics import create_dir
from . autoencoder import AutoEncoder
from .. models.point_net_based_AE import encoder, decoder

try:
    from tf_nndistance import nn_distance
except:
    pass


class Configuration():
    def __init__(self, n_input, n_z, training_epochs, batch_size, learning_rate=0.001,
                 saver_step=None, train_dir=None, loss='Bernoulli', gauss_augment=None,
                 z_rotate=False, non_linearity=tf.nn.relu, saver_max_to_keep=None,
                 loss_display_step=1, encoder=encoder, decoder=decoder):

        self.n_input = n_input
        self.n_z = n_z
        self.non_linearity = non_linearity
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
#         self.is_denoising = denoising
        self.z_rotate = z_rotate
        self.loss = loss
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decoder = decoder
        self.encoder = encoder


class VariationalAutoencoder(AutoEncoder):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and . The VAE can be learned end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, name, configuation):
        AutoEncoder.__init__(self, name)

        # Define the Tensor-Flow graph.
        c = configuation
        self.conf = c
        encoder = c.encoder
        decoder = c.decoder

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None] + c.n_input)
            self.z = self._encoded_to_latent(encoder(self.x))
            self.x_reconstr = decoder(self.z)

            if c.loss == 'Bernoulli':
                self.x_reconstr = tf.sigmoid(self.x_reconstr)   # Force Output to be in [0,1]

            # Define loss function based variational upper-bound and corresponding optimizer.
            self._create_loss_optimizer()

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self.init = tf.global_variables_initializer()

            # GPU configuration
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _encoded_to_latent(self, encoded):
        n_z = self.conf.n_z
        batch_size = self.conf.batch_size
        self.z_mean = fc_layer(encoded, n_z, activation='relu', weights_init='xavier')
        self.z_log_sigma_sq = fc_layer(encoded, n_z, activation='relu', weights_init='xavier')
        eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)  # TODO double check that this samples new stuff in each batch.
        # z = mu + sigma * epsilon
        return tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    def _create_loss_optimizer(self):
        if self.c.loss == 'Chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.x)
            reconstr_loss = tf.reduce_sum(cost_p1_p2) + tf.reduce_sum(cost_p2_p1)
        elif self.c.loss == 'Bernoulli':
            # Negative log probability of the input under the reconstructed Bernoulli distribution
            # induced by the decoder in the data space. Adding 1e-10 to avoid evaluation of log(0.0)
            reconstr_loss = - tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr) +
                                            (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr), 1)
        else:
            raise ValueError('Wrong loss was specified.')

        # Regularize posterior towards unit Gaussian prior:
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.loss = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss)

        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate).minimize(self.loss)

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})    # TODO _ think if z is a more reasonable transformation.

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=(self.conf.batch_size, self.conf.n_z))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr, feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr, feed_dict={self.x: X})

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_cost = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, _, _ = train_data.next_batch(batch_size)
            batch_i = batch_i.reshape([batch_size] + configuration.n_input)

            if configuration.loss == 'Bernoulli':
                batch_i_tmp = batch_i.copy()
                # Ensures pclouds lie in [0,1] interval, thus are interpreted as Bernoulli variables.
                batch_i_tmp += .5
                batch_i_tmp = np.maximum(1e-10, batch_i_tmp)
                batch_i = batch_i_tmp

            cost, _ = self.partial_fit(batch_i_tmp)
            # Compute average loss
            epoch_cost += cost

        epoch_cost /= (n_batches * batch_size)
        duration = time.time() - start_time
        return epoch_cost, duration

    def train(self, train_data, configuration):
        # Training cycle
        c = configuration
        if c.saver_step is not None:
            create_dir(c.train_dir)
        for epoch in xrange(c.training_epochs):
            cost, _ = self._single_epoch_train(train_data, c)
            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost))
            # Save the models checkpoint periodically.
            if c.saver_step is not None and epoch % c.saver_step == 0:
                checkpoint_path = osp.join(c.train_dir, 'models.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=epoch)
