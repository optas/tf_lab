'''
Created on January 26, 2017

@author: optas
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim

from .. fundamentals.layers import fully_connected_layer, relu, tanh, conv_1d
from .. fundamentals.loss import Loss 


class Configuration():
    def __init__(self, n_input, training_epochs, batch_size=10, learning_rate=0.001, transfer_fct=tf.nn.relu):

        self.n_input = n_input
        self.training_epochs = training_epochs
        self.encoder_sizes = [64, 64, 64, 128, 1024]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.transfer_fct = transfer_fct
        self.is_denoising = False
        self.padding = 'SAME'


class PointNetAutoEncoder(object):
    '''
    An Auto-Encoder replicating the architecture of Charles and Hao paper.
    '''

    def __init__(self, name, configuration):
        self.configuration = configuration
        c = self.configuration
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None, self.configuration.n_input, 3])
            layer = conv_1d(self.x, c.encoder_sizes[0], 1, 1, c.padding, name='conv-fc1')
            layer = conv_1d(layer, c.encoder_sizes[1], 1, 1, c.padding, name='conv-fc2')
            layer = conv_1d(layer, c.encoder_sizes[2], 1, 1, c.padding, name='conv-fc3')
            layer = conv_1d(layer, c.encoder_sizes[3], 1, 1, c.padding, name='conv-fc4')
            layer = conv_1d(layer, c.encoder_sizes[4], 1, 1, c.padding, name='conv-fc5')
            self.z = tf.reduce_max(layer, reduction_indices=1)
            layer = fully_connected_layer(self.z, self.configuration.n_input, name='decoder_fc_0')
            layer = fully_connected_layer(layer, self.configuration.n_input * 3, name='decoder_fc_1')
            self.x_reconstr = tf.reshape(layer, [-1, self.configuration.n_input, 3])

            if self.configuration.is_denoising:
                self.gt = tf.placeholder(tf.float32, [None, self.configuration.n_input, 3])    # The ground-truth (i.e. the denoised input).
            else:
                self.gt = self.x

        self._create_loss_optimizer()
        self.saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(init)

    def partial_fit(self, X, GT=None):
        '''Train model based on mini-batch of input data.
        Returns cost of mini-batch.'''
        if GT is not None:
            _, cost = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X, self.gt: GT})
        else:
            _, cost = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def reconstruct(self, X):
        '''Use AE to reconstruct given data.'''
        return self.sess.run((self.x_reconstr, self.loss), feed_dict={self.x: X, self.gt: X})

    def restore_model(self, model_path):
        #         r_vars = tf.trainable_variables()
        self.saver.restore(self.sess, model_path)

    def _encoder_network(self):
        # Generate encoder (recognition network), which maps inputs onto a latent space.
        c = self.configuration
        layer_sizes = c.encoder_sizes
        non_linearity = c.transfer_fct
        layer = non_linearity(fully_connected_layer(self.x, layer_sizes[0], stddev=0.01, name='encoder_fc_0'))
        layer = slim.batch_norm(layer,)

        for i in xrange(1, len(c.encoder_sizes)):
            layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], stddev=0.01, name='encoder_fc_' + str(i)))
            layer = slim.batch_norm(layer,)

        return layer

    def _decoder_network(self):
        # Generate a decoder which maps points from the latent space back onto the data space.
        c = self.configuration
        layer_sizes = c.decoder_sizes
        non_linearity = c.transfer_fct
        i = 0
        layer = non_linearity(fully_connected_layer(self.z, layer_sizes[i], stddev=0.01, name='decoder_fc_0'))
        layer = slim.batch_norm(layer,)
        for i in xrange(1, len(layer_sizes) - 1):
            layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], stddev=0.01, name='decoder_fc_' + str(i)))
            layer = slim.batch_norm(layer,)
        layer = fully_connected_layer(layer, layer_sizes[i + 1], stddev=0.01, name='decoder_fc_' + str(i + 1))  # Last decoding layer doesn't have a non-linearity.

        return layer

    def _create_loss_optimizer(self):
        c = self.configuration
        self.loss = Loss.l2_loss(self.x_reconstr, self.gt)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)
