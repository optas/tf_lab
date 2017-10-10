'''
Created on July 10, 2017

@author: optas
'''

import time
import tensorflow as tf
from tflearn import is_training

from . in_out import apply_augmentations
from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.inspect import count_trainable_parameters
from .. neural_net import NeuralNet


class PointNetClassifier(NeuralNet):
    '''
    An Classifier replicating the architecture of Charles and Hao paper.
    '''

    def __init__(self, name, configuration, graph=None):

        NeuralNet.__init__(self, name, graph)
        c = configuration
        self.configuration = c
        self.n_input = c.n_input
        self.n_output = c.n_output

        in_shape = [None] + self.n_input

        if c.one_hot:
            out_shape = [None] + self.n_output
        else:
            out_shape = [None]

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            self.gt = tf.placeholder(tf.int32, out_shape)
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
        try:
            is_training(True, session=self.sess)
            _, loss, prediction = self.sess.run((self.optimizer, self.loss, self.prediction), feed_dict={self.x: X, self.gt: GT})
            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
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
