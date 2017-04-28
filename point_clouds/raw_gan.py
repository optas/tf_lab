'''
Created on Apr 27, 2017

@author: optas
'''

import numpy as np
import time
import tensorflow as tf
from . encoders_decoders import decoder_with_fc_only_new
from tflearn.layers.core import fully_connected
from .. fundamentals.utils import expand_scope_by_name


class RawGAN():

    def __init__(self, learning_rate, n_output, noise_dim=128):

        self.noise_dim = noise_dim
        self.n_output = n_output
        out_shape = [None] + self.n_output
        self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])     # Noise vector.
        self.real_pc = tf.placeholder(tf.float32, out_shape)                 # Ground-truth.

        with tf.variable_scope('generator'):
            self.generator_out = self.generator(self.noise)

        with tf.variable_scope('discriminator') as scope:
            self.real_prob, self.real_logit = self.discriminator(self.real_pc, scope=scope)
            self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope)

        self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
        self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))

        train_vars = tf.trainable_variables()
        d_params = [v for v in train_vars if v.name.startswith('discriminator/')]
        g_params = [v for v in train_vars if v.name.startswith('generator/')]

        self.opt_d = self.optimizer(learning_rate, self.loss_d, d_params)
        self.opt_g = self.optimizer(learning_rate, self.loss_g, g_params)
        self.init = tf.global_variables_initializer()

        # Launch the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def generator(self, z, layer_sizes=[64, 128, 1024]):
        out_signal = decoder_with_fc_only_new(z, layer_sizes=layer_sizes)
        out_signal = fully_connected(out_signal, np.prod(self.n_output), activation='tanh', weights_init='xavier')
        out_signal = tf.reshape(out_signal, [-1, self.n_output[0], self.n_output[1]])
        return out_signal

    
    def discriminator(self, x, reuse=False, scope=None):
        
        name = 'conv_layer_0'
        expand_scope_by_name(scope, )
        
        layer = conv_1d(in_signal, nb_filter=layer_sizes[0], filter_size=1, strides=1, )

    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    layer = conv_1d(layer, nb_filter=layer_sizes[1], filter_size=1, strides=1, name='encoder_conv_layer_1')
    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    layer = conv_1d(layer, nb_filter=layer_sizes[2], filter_size=1, strides=1, name='encoder_conv_layer_2')
    if b_norm:
        layer = batch_normalization(layer)
    layer = non_linearity(layer)

    if dropout_prob is not None:
        layer = dropout(layer, dropout_prob)

    layer = symmetry(layer, axis=1)
        
        
        
        
        
        input_signal = x
        
        
        
        
        d_logits = decoder_with_fc_only_new(input_signal, layer_sizes=layer_sizes[:-1], reuse=reuse, scope=scope)

        if scope is not None:
            scope_i = scope.name + '/linear-mlp'
        else:
            scope_i = None

        d_logits = fully_connected(d_logits, layer_sizes[-1], activation='linear', weights_init='xavier', reuse=reuse, scope=scope_i)

        if scope is not None:
            scope_i = scope.name + '/single-logit'
        else:
            scope_i = None

        d_logit = fully_connected(d_logits, 1, activation='linear', weights_init='xavier', reuse=reuse, scope=scope_i)
        d_prob = tf.nn.sigmoid(d_logit)
        return d_prob, d_logit

    def generate(self, conditional_input, noise):
        feed_dict = {self.part_latent: conditional_input, self.z: noise}
        return self.sess.run([self.generator_out], feed_dict=feed_dict)

    def generator_noise_distribution(self, n_samples, ndims, mu=0, sigma=1):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def optimizer(self, learning_rate, loss, var_list):
        initial_learning_rate = learning_rate
        optimizer = tf.train.AdamOptimizer(initial_learning_rate, beta1=0.5).minimize(loss, var_list=var_list)
        return optimizer

    def _single_epoch_train(self, train_data, batch_size, sigma):
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
            z = self.generator_noise_distribution(batch_size, self.noise_dim, sigma=sigma)
            feed_dict = {self.real_pc: feed, self.z: z}
            loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)
            loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)

            # Compute average loss
            epoch_loss_d += loss_d
            epoch_loss_g += loss_g

        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
