'''
Created on May 3, 2017

@author: optas
'''

import os.path as osp
import warnings
import tensorflow as tf


from general_tools.in_out.basics import create_dir

from .. neural_net import Neural_Net


class GAN(Neural_Net):

    def __init__(self, name, graph):
        Neural_Net.__init__(self, name, graph)

    def save_model(self, tick):
        self.saver.save(self.sess, self.MODEL_SAVER_ID, global_step=tick)

    def optimizer(self, learning_rate, beta, loss, var_list):
        initial_learning_rate = learning_rate
        optimizer = tf.train.AdamOptimizer(initial_learning_rate, beta1=beta).minimize(loss, var_list=var_list)
        return optimizer

    def generate(self, n_samples, noise_params):
        noise = self.generator_noise_distribution(n_samples, self.noise_dim, **noise_params)
        feed_dict = {self.noise: noise}
        return self.sess.run([self.generator_out], feed_dict=feed_dict)[0]