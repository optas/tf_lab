'''
Created on Apr 27, 2017

@author: optas
'''

import numpy as np
import time
import tensorflow as tf

from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_1d
from tflearn.layers.normalization import batch_normalization

from . gan import GAN
from . encoders_decoders import decoder_with_fc_only_new
from .. fundamentals.utils import expand_scope_by_name


class LatentGAN(GAN):

    def __init__(self, name, learning_rate, n_output, noise_dim=128):

        self.noise_dim = noise_dim
        self.n_output = n_output
        out_shape = [None] + self.n_output

        GAN.__init__(self, name)      # TODO - push more sharable code in GAN class.

        with tf.variable_scope(name):

            self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])     # Noise vector.
            self.gt_data = tf.placeholder(tf.float32, shape=out_shape)           # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise)

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.gt_data, scope=scope)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope)

                self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
                self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))

                train_vars = tf.trainable_variables()

                d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
                g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]

                self.opt_d = self.optimizer(learning_rate, self.loss_d, d_params)
                self.opt_g = self.optimizer(learning_rate, self.loss_g, g_params)
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                self.init = tf.global_variables_initializer()

                # Launch the session
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                self.sess.run(self.init)

    def generator(self, z, layer_sizes=[64, 128]):
        layer_sizes = layer_sizes + self.n_output
        out_signal = decoder_with_fc_only_new(z, layer_sizes=layer_sizes, b_norm=False)
        out_signal = tf.nn.relu(out_signal)
        return out_signal

    def discriminator(self, in_signal, layer_sizes=[64, 128, 256, 512, 1024], reuse=False, scope=None):
        d_logits = decoder_with_fc_only_new(in_signal, layer_sizes=layer_sizes[:-1], reuse=reuse, scope=scope)
        name = 'single-logit'
        scope_e = expand_scope_by_name(scope, name)
        d_logit = fully_connected(d_logits, 1, activation='linear', weights_init='xavier', reuse=reuse, scope=scope_e)
        d_prob = tf.nn.sigmoid(d_logit)
        return d_prob, d_logit

    def generator_noise_distribution(self, n_samples, ndims, mu=0, sigma=1):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data, batch_size, noise_params):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        batch_size = batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for _ in xrange(n_batches):
            feed, _, _ = train_data.next_batch(batch_size)

            # Update discriminator.
            z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
            feed_dict = {self.gt_data: feed, self.noise: z}
            loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)
            loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)

            # Compute average loss
            epoch_loss_d += loss_d
            epoch_loss_g += loss_g

        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
