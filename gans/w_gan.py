'''
Author: Achlioptas Panos (Github ID: optas)
'''

import numpy as np
import time
import tensorflow as tf

from . gan import GAN

class W_GAN(GAN):
    '''TODO - make consistent with other classes.
    '''
    def __init__(self, name, learning_rate, clamp, n_output, noise_dim, discriminator, generator, beta=0.9, gen_kwargs={}, disc_kwargs={}):

        self.noise_dim = noise_dim
        self.n_output = n_output
        out_shape = [None] + self.n_output
        self.discriminator = discriminator
        self.generator = generator

        GAN.__init__(self, name)

        with tf.variable_scope(name):

            self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])     # Noise vector.
            self.real_pc = tf.placeholder(tf.float32, shape=out_shape)           # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.noise, self.n_output[0], **gen_kwargs)

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.real_pc, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope, **disc_kwargs)

            # Compute WGAN losses
            self.loss_d = tf.reduce_mean(self.synthetic_logit) - tf.reduce_mean(self.real_logit)
            self.loss_g = -tf.reduce_mean(self.synthetic_logit)

            train_vars = tf.trainable_variables()

            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]

            # Clip parameters of discriminator
            self.d_clipper = [p.assign(tf.clip_by_value(p, -clamp, clamp)) for p in d_params]

            self.opt_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss_d, var_list=d_params)
            self.opt_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss_g, var_list=g_params)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data, batch_size, noise_params, discriminator_boost=5):
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

        iterations_for_epoch = n_batches / discriminator_boost

        # Loop over all batches
        for _ in xrange(iterations_for_epoch):
            for _ in range(discriminator_boost):
                feed, _, _ = train_data.next_batch(batch_size)
                z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                feed_dict = {self.real_pc: feed, self.noise: z}
                _, _, loss_d = self.sess.run([self.opt_d, self.d_clipper, self.loss_d], feed_dict=feed_dict)
                epoch_loss_d += loss_d

            # Update generator.
            z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
            feed_dict = {self.noise: z}
            _, loss_g = self.sess.run([self.opt_g, self.loss_g], feed_dict=feed_dict)
            epoch_loss_g += loss_g

        epoch_loss_d /= (iterations_for_epoch * discriminator_boost)
        epoch_loss_g /= iterations_for_epoch
        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g), duration
