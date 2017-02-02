import tensorflow as tf
import tensorflow.contrib.slim as slim

from .. fundamentals.layers import fully_connected_layer, relu, tanh
from .. fundamentals.loss import Loss


class Configuration():
    def __init__(self, n_input, training_epochs, batch_size=10, learning_rate=0.001, transfer_fct=tf.nn.relu):

        self.n_input = n_input
        self.training_epochs = training_epochs
        self.encoder_sizes = [300, 200, 100]
        self.decoder_sizes = [200, 300, n_input]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.transfer_fct = transfer_fct
        self.is_denoising = False


class FullyConnectedAutoEncoder(object):
    '''
    A simple Auto-Encoder utilizing only Fully-Connected Layers.
    '''

    def __init__(self, name, configuration):
        self.configuration = configuration
        c = self.configuration
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None, c.n_input])
            self.z = self._encoder_network()
            self.x_reconstr = self._decoder_network()

            if self.configuration.is_denoising:
                self.gt = tf.placeholder(tf.float32, [None, c.n_input])    # The input has noise - need separate ground-truth placeholder.
            else:
                self.gt = self.x

        self._create_loss_optimizer()
        self.saver = tf.train.Saver(tf.global_variables())

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Launch the session
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(init)

    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)

    def _create_loss_optimizer(self):
        c = self.configuration
        self.cost = Loss.l2_loss(self.x_reconstr, self.gt)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.cost)

    def partial_fit(self, X, GT=None):
        '''Train models based on mini-batch of input data.
        Returns cost of mini-batch.
        If the AE is de-noising the GT needs to be provided.
        '''
        if GT is not None:
            _, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X, self.gt: GT})
        else:
            _, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def reconstruct(self, X):
        '''Use AE to reconstruct given data.'''
        return self.sess.run((self.x_reconstr, self.cost), feed_dict={self.x: X, self.gt: X})

    def _encoder_network(self):
        # Generate encoder (recognition network), which maps inputs onto a latent space.
        c = self.configuration
        layer_sizes = c.encoder_sizes
        non_linearity = c.transfer_fct
        layer = non_linearity(fully_connected_layer(self.x, layer_sizes[0], name='encoder_fc_0'))
        layer = slim.batch_norm(layer,)

        for i in xrange(1, len(c.encoder_sizes)):
            layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], name='encoder_fc_' + str(i)))
            layer = slim.batch_norm(layer,)

        return layer

    def _decoder_network(self):
        # Generate a decoder which maps points from the latent space back onto the data space.
        c = self.configuration
        layer_sizes = c.decoder_sizes
        non_linearity = c.transfer_fct
        i = 0
        layer = non_linearity(fully_connected_layer(self.z, layer_sizes[i], name='decoder_fc_0'))
        layer = slim.batch_norm(layer,)
        for i in xrange(1, len(layer_sizes) - 1):
            layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], name='decoder_fc_' + str(i)))
            layer = slim.batch_norm(layer,)
        layer = fully_connected_layer(layer, layer_sizes[i + 1], name='decoder_fc_' + str(i + 1))  # Last decoding layer doesn't have a non-linearity.

        return layer
