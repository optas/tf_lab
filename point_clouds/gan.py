'''
Created on May 3, 2017

@author: optas
'''

import os.path as osp
import warnings
import tensorflow as tf

from general_tools.in_out.basics import create_dir
from general_tools.python_oop import lazy_property

MODEL_SAVER_ID = 'models.ckpt'


class GAN(object):

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

        with tf.device('/cpu:0'), tf.name_scope(name):
            self.global_step = tf.get_variable('global_step', initializer=tf.constant_initializer(0), trainable=False)
            self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

    def save_model(self, tick):
        self.saver.save(self.sess, MODEL_SAVER_ID, global_step=tick)

    def restore_model(self, model_path, epoch, verbose=False):
        '''Restore all the variables of a saved model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, MODEL_SAVER_ID + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print('Model restored in epoch {0}.'.format(epoch))

    def optimizer(self, learning_rate, beta, loss, var_list):
        initial_learning_rate = learning_rate
        optimizer = tf.train.AdamOptimizer(initial_learning_rate, beta1=beta).minimize(loss, var_list=var_list)
        return optimizer

    def generate(self, n_samples, noise_params):
        noise = self.generator_noise_distribution(n_samples, self.noise_dim, **noise_params)
        feed_dict = {self.noise: noise}
        return self.sess.run([self.generator_out], feed_dict=feed_dict)[0]

    @lazy_property
    def saver(self, max_to_keep):
        return tf.train.Saver(self.all_variables(), max_to_keep=max_to_keep)

    @property
    def all_variables(self):
        tr_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.name)
        return sorted(tr_vars, key=lambda v: v.name)
