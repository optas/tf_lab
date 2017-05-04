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
# from tflearn.activations import leaky_relu

from . gan import GAN
from . encoders_decoders import decoder_with_fc_only_new
from .. fundamentals.utils import expand_scope_by_name


def leaky_relu(x, leak=0.3):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


class RawWGAN(GAN):

    def __init__(self, name, n_output, learning_rate=5e-5, clamp=0.01, noise_dim=128):

        self.noise_dim = noise_dim
        self.n_output = n_output
        out_shape = [None] + self.n_output

        GAN.__init__(self, name)      # TODO - push more sharable code in GAN class.

        with tf.variable_scope(name):

            self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])     # Noise vector.
            self.real_pc = tf.placeholder(tf.float32, shape=out_shape)           # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise)

            with tf.variable_scope('discriminator') as scope:
                self.real_logit = self.discriminator(self.real_pc, scope=scope)
                self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope)

                self.loss_d = -(tf.reduce_mean(self.real_logit) - tf.reduce_mean(self.synthetic_logit))
                self.loss_g = -tf.reduce_mean(self.synthetic_logit)

                train_vars = tf.trainable_variables()

                d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
                g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]

                # Clip parameters of discriminator
                self.d_clipper = [p.assign(tf.clip_by_value(p, -clamp, clamp)) for p in d_params]

#                 self.opt_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss_d, var_list=d_params)
#                 self.opt_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss_g, var_list=g_params)

                self.opt_d = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(self.loss_d, var_list=d_params)
                self.opt_g = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(self.loss_g, var_list=g_params)

#                 self.opt_d = self.optimizer(learning_rate, self.loss_d, d_params)
#                 self.opt_g = self.optimizer(learning_rate, self.loss_g, g_params)

                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                self.init = tf.global_variables_initializer()

                # Launch the session
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                self.sess.run(self.init)

    def generator(self, z, layer_sizes=[64, 128, 512, 1024]):
        # WGAN with bnorm true - didn't work so far.
        out_signal = decoder_with_fc_only_new(z, layer_sizes=layer_sizes, b_norm=False)
        out_signal = tf.nn.relu(out_signal)
        out_signal = fully_connected(out_signal, np.prod(self.n_output), activation='linear', weights_init='xavier')
        out_signal = tf.reshape(out_signal, [-1, self.n_output[0], self.n_output[1]])
        return out_signal

    def discriminator(self, in_signal, reuse=False, scope=None, leak=0.3):
        name = 'conv_layer_0'
        scope_e = expand_scope_by_name(scope, name)
        layer = conv_1d(in_signal, nb_filter=64, filter_size=1, strides=1, name=name, scope=scope_e, reuse=reuse)
        name += '_bnorm'
        scope_e = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, scope=scope_e, reuse=reuse)
        layer = tf.nn.relu(layer)
#         layer = leaky_relu(layer, leak)

        name = 'conv_layer_1'
        scope_e = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1, name=name, scope=scope_e, reuse=reuse)
        name += '_bnorm'
        scope_e = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, scope=scope_e, reuse=reuse)
        layer = tf.nn.relu(layer)
#         layer = leaky_relu(layer, leak)

        name = 'conv_layer_2'
        scope_e = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1, name=name, scope=scope_e, reuse=reuse)
        name += '_bnorm'
        scope_e = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, scope=scope_e, reuse=reuse)
        layer = tf.nn.relu(layer)
#         layer = leaky_relu(layer, leak)
        layer = tf.reduce_max(layer, axis=1)

        name = 'decoding_logits'
        scope_e = expand_scope_by_name(scope, name)
        d_logits = decoder_with_fc_only_new(layer, layer_sizes=[128, 64], reuse=reuse, scope=scope_e)

        name = 'single-logit'
        scope_e = expand_scope_by_name(scope, name)
        d_logit = fully_connected(d_logits, 1, activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_e)
        return d_logit

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

        discriminator_boost = 5
        iterations_for_epoch = n_batches / discriminator_boost

        # Loop over all batches
        for _ in xrange(iterations_for_epoch):
            for _ in range(discriminator_boost):
                feed, _, _ = train_data.next_batch(batch_size)
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                feed_dict = {self.real_pc: feed, self.noise: z}
                loss_d, _, _ = self.sess.run([self.loss_d, self.opt_d, self.d_clipper], feed_dict=feed_dict)
                epoch_loss_d += loss_d

            # Update generator.
#             z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
#             feed_dict = {self.noise: z}
            loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)
            epoch_loss_g += loss_g

        epoch_loss_d /= (iterations_for_epoch * discriminator_boost)
        epoch_loss_g /= iterations_for_epoch
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
