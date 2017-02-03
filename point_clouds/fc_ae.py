import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os.path as osp
import numpy as np

from general_tools.in_out.basics import create_dir
from . autoencoder import AutoEncoder
from .. fundamentals.layers import fully_connected_layer, relu, tanh
from .. fundamentals.loss import Loss


class Configuration():
    def __init__(self, n_input, training_epochs, batch_size, learning_rate=0.001, denoising=False,
                 saver_step=None, train_dir=None, transfer_fct=tf.nn.relu, loss_display_step=1):

        self.n_input = n_input
        self.training_epochs = training_epochs
        self.encoder_sizes = [300, 200, 100]
        self.decoder_sizes = [200, 300, n_input]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.transfer_fct = transfer_fct
        self.is_denoising = denoising
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir

    def __str__(self):
        keys = self.__dict__.keys()
        vals = self.__dict__.values()
        index = np.argsort(keys)
        res = ''
        for i in index:
            res += '%30s: %s\n' % (str(keys[i]), str(vals[i]))
        return res


class FullyConnectedAutoEncoder(AutoEncoder):
    '''
    A simple Auto-Encoder utilizing only Fully-Connected Layers.
    '''

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()

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

    def _create_loss_optimizer(self):
        c = self.configuration
        self.cost = Loss.l2_loss(self.x_reconstr, self.gt)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.cost)

    def _encoder_network(self):
        # Generate encoder (recognition network), which maps inputs onto a latent space.
        c = self.configuration
        layer_sizes = c.encoder_sizes
        non_linearity = c.transfer_fct
        layer = non_linearity(fully_connected_layer(self.x, layer_sizes[0], name='encoder_fc_0'))
        layer = slim.batch_norm(layer)

        for i in xrange(1, len(c.encoder_sizes)):
            layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], name='encoder_fc_' + str(i)))
            layer = slim.batch_norm(layer)

        return layer

    def _decoder_network(self):
        # Generate a decoder which maps points from the latent space back onto the data space.
        c = self.configuration
        layer_sizes = c.decoder_sizes
        non_linearity = c.transfer_fct
        i = 0
        layer = non_linearity(fully_connected_layer(self.z, layer_sizes[i], name='decoder_fc_0'))
        layer = slim.batch_norm(layer)
        for i in xrange(1, len(layer_sizes) - 1):
            layer = non_linearity(fully_connected_layer(layer, layer_sizes[i], name='decoder_fc_' + str(i)))
            layer = slim.batch_norm(layer,)
        layer = fully_connected_layer(layer, layer_sizes[i + 1], name='decoder_fc_' + str(i + 1))  # Last decoding layer doesn't have a non-linearity.

        return layer

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_cost = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, _, _ = train_data.next_batch(batch_size)
            cost, _ = self.partial_fit(batch_i)
            # Compute average loss
            epoch_cost += cost

        epoch_cost /= n_batches
#                        * batch_size)
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
