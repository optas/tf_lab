'''
Created on Mar 1, 2017

@author: optas
'''

from . import PointNetAutoEncoder
import time
import numpy as np
import tensorflow as tf


import time
from . autoencoder import AutoEncoder


class PNetAE_with_Boosted_Decoder(AutoEncoder, PointNetAutoEncoder):
    '''
    classdocs
    '''

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()
        self.configuration = configuration
        c = self.configuration

        AutoEncoder.__init__(self, name, c.n_input, c.is_denoising)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)

            booster_layer = fully_connected(c.booster_dim, 32, activation='relu', weights_init='xavier')
            layer = tf.concat(1, [self.z, booster_layer])
            layer = c.decoder(layer, **c.decoder_args)

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

    def _single_epoch_train(self, train_data, configuration):
            n_examples = train_data.num_examples
            epoch_loss = 0.
            batch_size = configuration.batch_size
            n_batches = int(n_examples / batch_size)
            start_time = time.time()
            # Loop over all batches
            for _ in xrange(n_batches):
                if self.is_denoising:
                    original_data, _, batch_i = train_data.next_batch(batch_size)
                    if batch_i is None:  # In this case the denoising concern only the augmentation.
                        batch_i = original_data
                else:
                    batch_i, _, _ = train_data.next_batch(batch_size)

                batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

                if self.is_denoising:
                    loss, _ = self.partial_fit(batch_i, original_data)
                else:
                    loss, _ = self.partial_fit(batch_i)

                # Compute average loss
                epoch_loss += loss
            epoch_loss /= n_batches
            duration = time.time() - start_time
            return epoch_loss, duration