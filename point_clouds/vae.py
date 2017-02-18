'''
Created on February 1, 2017

@author: optas
'''

import numpy as np
import time
import tensorflow as tf
from tflearn.layers.core import fully_connected as fc_layer

from general_tools.in_out.basics import create_dir
from general_tools.rla.three_d_transforms import rand_rotation_matrix
from general_tools.in_out.basics import pickle_data

from tensorflow.contrib.losses import compute_weighted_loss

try:
    from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
    from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print 'External Losses (Chamfer-EMD) cannot be loaded.'

from . autoencoder import AutoEncoder


class VariationalAutoencoder(AutoEncoder):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and . The VAE can be learned end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()

        self.configuration = configuration
        c = self.configuration
        AutoEncoder.__init__(self, name, c.n_input, c.is_denoising)

        encoder = c.encoder
        decoder = c.decoder

        with tf.variable_scope(name):
            self.z = self._encoded_to_latent(encoder(self.x))
            self.x_reconstr = decoder(self.z)

            if c.loss == 'bernoulli':
                self.x_reconstr = tf.sigmoid(self.x_reconstr)   # Force Output to be in [0,1]

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)
            # Define loss function based variational upper-bound and corresponding optimizer.
            self._create_loss_optimizer()

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # GPU configuration
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _encoded_to_latent(self, encoded):
        n_z = self.configuration.n_z
        batch_size = self.configuration.batch_size
        self.z_mean = fc_layer(encoded, n_z, activation='relu', weights_init='xavier')
        self.z_log_sigma_sq = fc_layer(encoded, n_z, activation='relu', weights_init='xavier')
        eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)  # TODO double check that this samples new stuff in each batch.
        # z = mu + sigma * epsilon
        return tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    def _create_loss_optimizer(self):
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            reconstr_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            reconstr_loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))
        elif c.loss == 'bernoulli':
            # Negative log probability of the input under the reconstructed Bernoulli distribution
            # induced by the decoder in the data space. Adding 1e-10 to avoid evaluation of log(0.0)
            reconstr_loss = - tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr) +
                                            (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr), 1)
        else:
            raise ValueError('Wrong loss was specified.')

        # Regularize posterior towards unit Gaussian prior:
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)
	
        self.loss = tf.reduce_mean(reconstr_loss + latent_loss)   # TODO - >add weighted loss
	#self.loss = compute_weighted_loss(temp, [1.0, c.latent_vs_recon])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)

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
            z_mu = np.random.normal(size=(self.configuration.batch_size, self.configuration.n_z))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr, feed_dict={self.z: z_mu})

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        for _ in xrange(n_batches):
            original_data, labels, noisy_data = train_data.next_batch(batch_size)

            if self.configuration.debug and not set(labels).issubset(self.train_names):
                assert(False)

            original_data = original_data.reshape([batch_size] + configuration.n_input)

            if self.is_denoising:
                noisy_data = noisy_data.reshape([batch_size] + configuration.n_input)
                batch_i = noisy_data
            else:
                batch_i = original_data

            batch_i_tmp = batch_i.copy()    # TODO -> only necessary if you do augmentations

            if configuration.gauss_augment is not None:
                mu = configuration.gauss_augment['mu']
                sigma = configuration.gauss_augment['sigma']
                batch_i_tmp += np.random.normal(mu, sigma, batch_i_tmp.shape)

            if configuration.z_rotate:  # TODO -> add independent rotations to each object
                r_rotation = rand_rotation_matrix()
                r_rotation[0, 2] = 0
                r_rotation[2, 0] = 0
                r_rotation[1, 2] = 0
                r_rotation[2, 1] = 0
                r_rotation[2, 2] = 1
                batch_i = batch_i.dot(r_rotation)

            if configuration.loss == 'bernoulli':
                # Ensures pclouds lie in [0,1] interval, thus are interpreted as Bernoulli variables.
                batch_i_tmp += .5
                batch_i_tmp = np.maximum(1e-10, batch_i_tmp)
                batch_i_tmp = np.minimum(batch_i_tmp, 1.0 - 1e-10)
                if np.max(batch_i_tmp) > 1 or np.min(batch_i_tmp) < 0:
                    print '%.10f' % (np.max(batch_i_tmp))
                    print '%.10f' % (np.min(batch_i_tmp))
                    raise ValueError()

            if self.is_denoising:
                loss, _ = self.partial_fit(batch_i_tmp, original_data)
            else:
                loss, _ = self.partial_fit(batch_i_tmp)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration
