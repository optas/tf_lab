import tensorflow as tf
import numpy as np

from .. fundamentals.layers import fully_connected_layer, relu, tanh
from .. fundamentals.loss import Loss 


class Configuration():
    def __init__(self, n_input, training_epochs, batch_size=10, learning_rate=0.001, transfer_fct=tf.nn.relu):

        self.n_input = n_input
        self.training_epochs = training_epochs
        self.hidden_out_sizes = [n_input,
                                 int(np.around(0.5 * n_input)),
                                 int(np.around(0.1 * n_input)),
                                 int(np.around(0.5 * n_input)),
                                 n_input\
                                 ]

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.transfer_fct = transfer_fct


class FullyConnectedAutoEncoder(object):
    '''
    A simple Auto-Encoder utilizing only Fully-Connected Layers.
    '''

    def __init__(self, configuration):
        self.configuration = configuration

        self.x = tf.placeholder(tf.float32, [None, self.configuration.n_input])
        self.z = self._encoder_network()
        self.x_reconstr = self._decoder_network()

        self._create_loss_optimizer()
        self.saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def partial_fit(self, X):
        '''Train model based on mini-batch of input data.
        Returns cost of mini-batch.'''
        _, cost = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def reconstruct(self, X):
        '''Use AE to reconstruct given data.'''
        return self.sess.run(self.x_reconstr, feed_dict={self.x: X})

    def restore_model(self, model_path):
        #         r_vars = tf.trainable_variables()
        self.saver.restore(self.sess, model_path)

    def _encoder_network(self):
        # Generate encoder (recognition network), which maps inputs onto a latent space.
        c = self.configuration
        layer = fully_connected_layer(self.x, c.hidden_out_sizes[0], stddev=0.01, name='fc_1')
        layer = fully_connected_layer(c.transfer_fct(layer), c.hidden_out_sizes[1], stddev=0.01, name='fc_2')
        return fully_connected_layer(c.transfer_fct(layer), c.hidden_out_sizes[2], stddev=0.01, name='fc_3')

    def _decoder_network(self):
        # Generate a decoder which maps points from the latent space back onto the data space.
        c = self.configuration
        layer = fully_connected_layer(c.transfer_fct(self.z), c.hidden_out_sizes[3], stddev=0.01, name='fc_4')
        layer = fully_connected_layer(layer, c.hidden_out_sizes[4], stddev=0.01, name='fc_5')
        return tanh(layer)

    def _create_loss_optimizer(self):
        c = self.configuration
        self.loss = Loss.l2_loss(self.x_reconstr, self.x)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)


def autoencoder_with_fcs_only(in_signal, configuration):
    hs = configuration.hidden_out_sizes
    n = configuration.n_points
    e = configuration.original_embedding
    transfer_fct = configuration.transfer_fct

    if e > 1:
        in_signal = tf.reshape(in_signal, [-1, n * e])
    layer = fully_connected_layer(in_signal, hs[0], stddev=0.01, name='fc_1')
    layer = fully_connected_layer(transfer_fct(layer), hs[1], stddev=0.01, name='fc_2')
    layer = fully_connected_layer(transfer_fct(layer), hs[2], stddev=0.01, name='fc_3')
    layer = fully_connected_layer(transfer_fct(layer), hs[3], stddev=0.01, name='fc_4')
    layer = fully_connected_layer(layer, hs[4], stddev=0.01, name='fc_5')
    layer = tanh(layer)
    if e > 1:
        layer = tf.reshape(layer, [-1, n, e])
    return layer
