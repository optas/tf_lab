'''
Created on May 5, 2017

@author: optas
'''

import numpy as np
import time
import tensorflow as tf

from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_1d
from tflearn.layers.normalization import batch_normalization

from general_tools.simpletons import iterate_in_chunks
from . gan import GAN
from . encoders_decoders import decoder_with_fc_only_new
from .. fundamentals.utils import expand_scope_by_name


class SharpenGAN(GAN):

    def __init__(self, name, in_gan, in_decoder, n_output, learning_rate=5e-5):
        self.synthetic_pc = tf.placeholder(tf.float32, shape=[None, n_output])
        self.real_pc = tf.placeholder(tf.float32, shape=[None, n_output])

        with tf.variable_scope('sharpen_discriminator') as scope:
            self.real_prob, _ = self.discriminator(self.real_pc, scope=scope)
            self.synthetic_prob, _ = self.discriminator(self.synthetic_pc, reuse=True, scope=scope)

        self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
        self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))

        train_vars = tf.trainable_variables()
        d_params = [v for v in train_vars if v.name.startswith(name + '/sharpen_discriminator/')]

#         g_params = [v for v in train_vars if v.name.startswith(name + '/sharpen_generator/')]
        output_z_sg = tf.stop_gradient(self.decoder.z)
        self.generator_out = self.generator(self.noise)
        self.loss_g

        self.opt_d = self.optimizer(learning_rate, self.loss_d, d_params)
        self.opt_g = self.optimizer(learning_rate, self.loss_g, g_params)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.init = tf.global_variables_initializer()

        # Launch the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def discriminator(self, in_signal, reuse=False, scope=None):
        name = 'conv_layer_0'
        scope_e = expand_scope_by_name(scope, name)
        layer = conv_1d(in_signal, nb_filter=64, filter_size=1, strides=1, name=name, scope=scope_e, reuse=reuse)
        name += '_bnorm'
        scope_e = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, scope=scope_e, reuse=reuse)
        layer = tf.nn.relu(layer)

        name = 'conv_layer_1'
        scope_e = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=128, filter_size=1, strides=1, name=name, scope=scope_e, reuse=reuse)
        name += '_bnorm'
        scope_e = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, scope=scope_e, reuse=reuse)
        layer = tf.nn.relu(layer)

        name = 'conv_layer_2'
        scope_e = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=1024, filter_size=1, strides=1, name=name, scope=scope_e, reuse=reuse)
        name += '_bnorm'
        scope_e = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, scope=scope_e, reuse=reuse)
        layer = tf.nn.relu(layer)
        layer = tf.reduce_max(layer, axis=1)

        name = 'decoding_logits'
        scope_e = expand_scope_by_name(scope, name)
        # try to remove bnorm here
        d_logits = decoder_with_fc_only_new(layer, layer_sizes=[128, 64], reuse=reuse, scope=scope_e)

        name = 'single-logit'
        scope_e = expand_scope_by_name(scope, name)
        d_logit = fully_connected(d_logits, 1, activation='linear', weights_init='xavier', name=name, reuse=reuse, scope=scope_e)
        d_prob = tf.nn.sigmoid(d_logit)
        return d_prob, d_logit

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

        synthetic_pc = self.in_decoder.decode(self.in_gan.generate(n_examples, noise_params))
        synthetic_iterator = iterate_in_chunks(synthetic_pc)

        # Loop over all batches
        for _ in xrange(n_batches):
            feed, _, _ = train_data.next_batch(batch_size)
            syn_feed = synthetic_iterator.next(batch_size)

            # Update discriminator.
            feed_dict = {self.real_pc: feed, self.synthetic_pc: syn_feed}
            loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)

            # Update decoder.
            loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)

            # Compute average loss
            epoch_loss_d += loss_d
            epoch_loss_g += loss_g

        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
