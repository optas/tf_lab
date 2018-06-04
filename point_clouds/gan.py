'''
Created on May 3, 2017

@author: optas
'''

import os.path as osp
import warnings
import tensorflow as tf


from general_tools.in_out.basics import create_dir
from .. neural_net import Neural_Net


def safe_log(x, eps=1e-12):
    return tf.log(tf.maximum(x, eps))


class GAN(Neural_Net):

    def __init__(self, name, graph):
        Neural_Net.__init__(self, name, graph)

    def optimizer(self, learning_rate, beta, loss, var_list):
        initial_learning_rate = learning_rate
        optimizer = tf.train.AdamOptimizer(initial_learning_rate, beta1=beta).minimize(loss, var_list=var_list)
        return optimizer

    def generate(self, n_samples, noise_params):
        noise = self.generator_noise_distribution(n_samples, self.noise_dim, **noise_params)
        feed_dict = {self.noise: noise}
        return self.sess.run([self.generator_out], feed_dict=feed_dict)[0]

    def vanilla_gan_objective(self, real_prob, synthetic_prob, safe_log=True):
        if safe_log:
            log = safe_log
        else:
            log = tf.log

        loss_d = tf.reduce_mean(-log(real_prob) - log(1 - synthetic_prob))
        loss_g = tf.reduce_mean(-log(synthetic_prob))
        return loss_d, loss_g

    def w_gan_objective(self, real_logit, synthetic_logit):
        loss_d = tf.reduce_mean(synthetic_logit) - tf.reduce_mean(real_logit)
        loss_g = -tf.reduce_mean(synthetic_logit)
        return loss_d, loss_g
