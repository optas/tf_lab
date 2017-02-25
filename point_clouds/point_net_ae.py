'''
Created on January 26, 2017

@author: optas
'''

import time
import numpy as np
import tensorflow as tf

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from general_tools.in_out.basics import create_dir
from general_tools.rla.three_d_transforms import rand_rotation_matrix


from . autoencoder import AutoEncoder
from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.loss import Loss

try:
    from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
    from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print 'External Losses (Chamfer-EMD) cannot be loaded.'


class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder replicating the architecture of Charles and Hao paper.
    '''

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()
        self.configuration = configuration
        c = self.configuration

        AutoEncoder.__init__(self, name, c.n_input, c.is_denoising)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            layer = c.decoder(self.z, **c.decoder_args)
            self.x_reconstr = tf.reshape(layer, [-1, c.n_input[0], c.n_input[1]])

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)
            self._create_loss_optimizer()

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # GPU configuration
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_loss_optimizer(self):
        c = self.configuration
        if c.loss == 'l2':
            self.loss = Loss.l2_loss(self.x_reconstr, self.gt)
        elif c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        for _ in xrange(n_batches):
            original_data, labels, noisy_data = train_data.next_batch(batch_size)

#             original_data = original_data.reshape([batch_size] + configuration.n_input)

            if self.is_denoising:
#                 noisy_data = noisy_data.reshape([batch_size] + configuration.n_input)
                batch_i = noisy_data
            else:
                batch_i = original_data

            batch_i = batch_i.copy()    # TODO -> only necessary if you do augmentations

            if configuration.gauss_augment is not None:
                mu = configuration.gauss_augment['mu']
                sigma = configuration.gauss_augment['sigma']
                batch_i += np.random.normal(mu, sigma, batch_i.shape)

            if configuration.z_rotate:  # TODO -> add independent rotations to each object
                r_rotation = rand_rotation_matrix()
                r_rotation[0, 2] = 0
                r_rotation[2, 0] = 0
                r_rotation[1, 2] = 0
                r_rotation[2, 1] = 0
                r_rotation[2, 2] = 1
                batch_i = batch_i.dot(r_rotation)

            if self.is_denoising:
                loss, _ = self.partial_fit(batch_i, original_data)
            else:
                loss, _ = self.partial_fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration
