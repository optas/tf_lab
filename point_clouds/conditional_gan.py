'''
Created on Apr 11, 2017

@author: optas
'''

import numpy as np
import time
import tensorflow as tf
from . encoders_decoders import decoder_with_fc_only


class ConditionalGAN():

    def __init__(self, n_gt_latent=1024, n_part_latent=1024, noise_dim=128):
        self.noise_dim = noise_dim
        self.z = tf.placeholder(tf.float32, shape=[None, noise_dim])                  # Noise vector.
        self.part_latent = tf.placeholder(tf.float32, shape=[None, n_part_latent])    # Latent code of part.
        self.gt_latent = tf.placeholder(tf.float32, shape=[None, n_gt_latent])      # Latent code of full shape.

        with tf.variable_scope('generator'):
            self.generator_out = self.conditional_generator(self.z, self.part_latent)

        with tf.variable_scope('discriminator') as scope:
            self.real_prob, _ = self.conditional_discriminator(self.gt_latent, self.part_latent, scope=scope)
            self.synthetic_prob, _ = self.conditional_discriminator(self.generator_out, self.part_latent, reuse=True, scope=scope)

        self.loss_d = tf.reduce_mean(-tf.log(self.real_prob) - tf.log(1 - self.synthetic_prob))
        self.loss_g = tf.reduce_mean(-tf.log(self.synthetic_prob))

        train_vars = tf.trainable_variables()
        d_params = [v for v in train_vars if v.name.startswith('discriminator/')]
        g_params = [v for v in train_vars if v.name.startswith('generator/')]

        self.opt_d = self.optimizer(self.loss_d, d_params)
        self.opt_g = self.optimizer(self.loss_g, g_params)
        self.init = tf.global_variables_initializer()

        # Launch the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu=0, sigma=1):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def conditional_generator(self, z, y, layer_sizes=[64, 128, 1024]):
        '''Given y and noise (z) generate data.'''
        input_signal = tf.concat(concat_dim=1, values=[z, y])
        out_signal = decoder_with_fc_only(input_signal, layer_sizes=layer_sizes)
        return out_signal

    def conditional_discriminator(self, x, y, layer_sizes=[128, 256, 512, 1024], reuse=False, scope=None):
        '''Decipher if input x is real or fake given y.'''
        input_signal = tf.concat(concat_dim=1, values=[x, y])
        d_logits = decoder_with_fc_only(input_signal, layer_sizes=layer_sizes, reuse=reuse, scope=scope)
        d_prob = tf.nn.sigmoid(d_logits)
        return d_prob, d_logits

    def optimizer(self, loss, var_list):
        initial_learning_rate = 0.005
        decay = 0.95
        num_decay_steps = 100
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate,
            batch,
            num_decay_steps,
            decay,
            staircase=True
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch, var_list=var_list)
        return optimizer

    def _single_epoch_train(self, train_data, batch_size):
        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        batch_size = batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        for _ in xrange(n_batches):
            gt_latent, _, part_latent = train_data.next_batch(batch_size)
            z = self.generator_noise_distribution(batch_size, self.noise_dim)
            feed_dict = {self.part_latent: part_latent, self.gt_latent: gt_latent, self.z: z}

            # Update discriminator.
            loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict=feed_dict)

            # Update generator.
            z = self.generator_noise_distribution(batch_size, self.noise_dim)
            feed_dict = {self.part_latent: part_latent, self.gt_latent: gt_latent, self.z: z}
            loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict=feed_dict)

            # Compute average loss
            epoch_loss_d += loss_d
            epoch_loss_g += loss_g

        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
