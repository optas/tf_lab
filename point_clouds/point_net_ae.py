'''
Created on January 26, 2017

@author: optas
'''

import time
import os.path as osp
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import conv_1d

from general_tools.in_out.basics import create_dir

from . autoencoder import AutoEncoder
from .. fundamentals.layers import fully_connected_layer, relu, tanh
from .. fundamentals.loss import Loss 


class Configuration():
    def __init__(self, n_input, training_epochs, batch_size=10, learning_rate=0.001, denoising=False, transfer_fct=tf.nn.relu,
                 saver_step=None, train_dir=None, loss_display_step=1):

        self.n_input = n_input
        self.training_epochs = training_epochs
        self.encoder_sizes = [64, 128, 1024]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.transfer_fct = transfer_fct
        self.is_denoising = denoising
        self.padding = 'SAME'
        self.is_denoising = denoising
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir


class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder replicating the architecture of Charles and Hao paper.
    '''

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()

        self.configuration = configuration
        c = self.configuration
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None, c.n_input[0], c.n_input[1]])
            self.z = self._encoder_network()
            self.x_reconstr = self._decoder_network()

            if self.configuration.is_denoising:
                self.gt = tf.placeholder(tf.float32, [None, c.n_input[0], c.n_input[1]])
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

    def _encoder_network(self):
        # Generate encoder (recognition network), which maps inputs onto a latent space.
        c = self.configuration
        layer = conv_1d(self.x, c.encoder_sizes[0], 1, 1, c.padding, name='conv-fc1')
        layer = conv_1d(layer, c.encoder_sizes[1], 1, 1, c.padding, name='conv-fc2')
        layer = conv_1d(layer, c.encoder_sizes[2], 1, 1, c.padding, name='conv-fc3')
        tf.reduce_max(layer, axis=1)
        return layer

    def _decoder_network(self):
        # Generate a decoder which maps points from the latent space back onto the data space.
        c = self.configuration
        layer = fully_connected_layer(self.z, c.n_input[0], name='decoder_fc_0')
        layer = slim.batch_norm(layer)
        layer = fully_connected_layer(layer, c.n_input[0] * c.n_input[1], name='decoder_fc_1')
        layer = slim.batch_norm(layer)
        return tf.reshape(layer, [-1, c.n_input[0], c.n_input[1]])

    def _create_loss_optimizer(self):
        c = self.configuration
        self.loss = Loss.l2_loss(self.x_reconstr, self.gt)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, _, _ = train_data.next_batch(batch_size)
            batch_i = batch_i.reshape([batch_size] + configuration.n_input)
            loss, _ = self.partial_fit(batch_i)
            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
    #                        * batch_size)
        duration = time.time() - start_time
        return epoch_loss, duration

    def train(self, train_data, configuration):
        # Training cycle
        c = configuration
        if c.saver_step is not None:
            create_dir(c.train_dir)
        for epoch in xrange(c.training_epochs):
            loss, _ = self._single_epoch_train(train_data, c)
            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(loss))
            # Save the models checkpoint periodically.
            if c.saver_step is not None and epoch % c.saver_step == 0:
                checkpoint_path = osp.join(c.train_dir, 'models.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=epoch)
