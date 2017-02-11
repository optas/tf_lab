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

try:
    from tf_nndistance import nn_distance
except:
    pass

from . autoencoder import AutoEncoder
from . spatial_transformer import transformer as pcloud_spn

from .. fundamentals.loss import Loss


class Configuration():
    def __init__(self, n_input, training_epochs, batch_size=10, learning_rate=0.001, denoising=False, non_linearity=tf.nn.relu,
                 saver_step=None, train_dir=None, z_rotate=False, loss='l2', gauss_augment=None, decoder=None, encoder=None, saver_max_to_keep=None, loss_display_step=1,
                 spatial_trans=False, debug=False):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs

        self.n_input = n_input
        self.encoder_sizes = [64, 128, 1024]
        self.non_linearity = non_linearity
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.spatial_trans = spatial_trans
        self.debug = debug

        def __str__(self):
            keys = self.__dict__.keys()
            vals = self.__dict__.values()
            index = np.argsort(keys)
            res = ''
            for i in index:
                res += '%30s: %s\n' % (str(keys[i]), str(vals[i]))
            return res


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
            if c.encoder is None:
                self.z = self._encoder_network(c.spatial_trans)
            else:
                self.z = c.encoder(self.x, c.spatial_trans)

            if c.decoder is None:
                self.x_reconstr = self._decoder_network()
            else:
                self.x_reconstr = c.decoder(self.z)

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

    def _encoder_network(self, spn=False):
        '''Generate encoder (recognition network), which maps inputs onto a latent space.
        '''
        c = self.configuration
        nb_filters = c.encoder_sizes
        if spn:
            layer = pcloud_spn(self.x)
        else:
            layer = self.x
        layer = conv_1d(layer, nb_filter=nb_filters[0], filter_size=1, strides=1, name='conv-fc1')
        layer = c.non_linearity(layer)
        layer = conv_1d(layer, nb_filter=nb_filters[1], filter_size=1, strides=1, name='conv-fc2')
        layer = c.non_linearity(layer)
        layer = conv_1d(layer, nb_filter=nb_filters[2], filter_size=1, strides=1, name='conv-fc3')
        layer = tf.reduce_max(layer, axis=1)
        return layer

    def _decoder_network(self):
        '''Generate a decoder which maps points from the latent space back onto the data space.
        '''
        c = self.configuration
        layer = fully_connected(self.z, c.n_input[0], name='decoder_fc_0')
        layer = c.non_linearity(layer)
        layer = fully_connected(layer, c.n_input[0] * c.n_input[1], name='decoder_fc_1')
        layer = tf.reshape(layer, [-1, c.n_input[0], c.n_input[1]])
        return layer

    def _create_loss_optimizer(self):
        c = self.configuration
        if c.loss == 'l2':
            self.loss = Loss.l2_loss(self.x_reconstr, self.gt)
        elif c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

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

            if self.configuration.debug and not set(labels).issubset(self.train_names):
                assert(False)

            original_data = original_data.reshape([batch_size] + configuration.n_input)

            if self.is_denoising:
                noisy_data = noisy_data.reshape([batch_size] + configuration.n_input)
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
