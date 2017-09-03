'''
Created on July 10, 2017

@author: optas
'''

import time
import tensorflow as tf
from . in_out import apply_augmentations
from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.inspect import count_trainable_parameters


class PointNetClassifier(object):
    '''
    An Classifier replicating the architecture of Charles and Hao paper.
    '''

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()

        c = configuration
        self.configuration = c
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        in_shape = [None] + self.n_input

        if c.one_hot:
            out_shape = [None] + self.n_output
        else:
            out_shape = [None]

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            self.gt = tf.placeholder(tf.int32, out_shape)
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

            self.z = c.encoder(self.x, **c.encoder_args)
            self.prediction = c.decoder(self.z, **c.decoder_args)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)
            self._create_loss_optimizer()

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

    def trainable_parameters(self):
        # TODO: what happens if more nets in single graph?
        return count_trainable_parameters(self.graph)

    def _create_loss_optimizer(self):
        c = self.configuration
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.gt)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('Classification_loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)

    def partial_fit(self, X, GT):
        '''Trains the model with mini-batches of input data.'''
        _, loss, prediction = self.sess.run((self.optimizer, self.loss, self.prediction), feed_dict={self.x: X, self.gt: GT})
        return prediction, loss

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, labels, _ = train_data.next_batch(batch_size)
            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.
            _, loss = self.partial_fit(batch_i, labels)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration
