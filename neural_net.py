'''
Created on August 28, 2017

@author: optas
'''

import numpy as np
import tensorflow as tf
import warnings
import os.path as osp
from six import itervalues, iterkeys

from fundamentals.inspect import count_trainable_parameters
from general_tools.in_out.basics import pickle_data, unpickle_data


MODEL_SAVER_ID = 'models.ckpt'


class Neural_Net(object):

    def __init__(self, name, graph, seed=None):
        if graph is None:
            graph = tf.get_default_graph()
        
        self.graph = graph
        self.name = name
        
        if seed is not None:        
            tf.set_random_seed(seed)
            
        with self.graph.as_default():
            with tf.variable_scope(name):
                with tf.device('/cpu:0'):
                    self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

                self.increment_epoch = self.epoch.assign_add(tf.constant(1.0))

            self.no_op = tf.no_op()

    def is_training(self):
        is_training_op = self.graph.get_collection('is_training')
        return self.sess.run(is_training_op)[0]

    def trainable_parameters(self):
        return count_trainable_parameters(self.graph, name_space=self.name)

    def restore_model(self, model_path, epoch, verbose=False):
        '''Restore all the variables of a saved model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, MODEL_SAVER_ID + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print('Model restored in epoch {0}.'.format(epoch))

    def start_session(self, allow_growth=True, random_seed=None):
        '''Associates with the Neura_Net a tf.Session object which is used to initialize the tf.global_variables'''
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = allow_growth

        if random_seed is not None:
            tf.set_random_seed(random_seed)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(self.init)


class Neural_Net_Conf(object):
    def __init__(self):
        pass

    def value_or_none(self, attribute):
        if hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            return None

    def __str__(self):
        keys = list(iterkeys(self.__dict__))
        vals = list(itervalues(self.__dict__))
        index = np.argsort(keys)
        res = ''
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__ + ' # ' + vals[i].func_code.co_filename
            else:
                v = str(vals[i])
            res += '%30s: %s\n' % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + '.pickle', self)
        with open(file_name + '.txt', 'w') as fout:
            fout.write(self.__str__())
            
    @staticmethod
    def load(file_name):
        return unpickle_data(file_name + '.pickle').next()
