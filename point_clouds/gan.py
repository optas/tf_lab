'''
Created on May 3, 2017

@author: optas
'''

import os.path as osp
import warnings
import numpy as np
import tensorflow as tf

from general_tools.in_out.basics import create_dir


model_saver_id = 'models.ckpt'


class GAN(object):
    '''
    classdocs
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name
        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

    def generate(self, noise):
        feed_dict = {self.noise: noise}
        return self.sess.run([self.generator_out], feed_dict=feed_dict)

    def generate_many(self, n_samples, sigma=1):
        gen_samples = np.zeros(([n_samples] + self.n_output))
        for i in xrange(n_samples):
            pc_gen = self.generate(self.generator_noise_distribution(1, self.noise_dim, sigma=sigma))[0][0]
            gen_samples[i] = pc_gen
        return gen_samples

    def restore_model(self, model_path, epoch, verbose=False):
        '''Restore all the variables of a saved model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, model_saver_id + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print('Model restored in epoch {0}.'.format(epoch))
