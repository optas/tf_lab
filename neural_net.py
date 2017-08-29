'''
Created on Aug 28, 2017

@author: optas
'''
from macerrors import sessClosedErr

MODEL_SAVER_ID = 'models.ckpt'

import os.path as osp
import tensorflow as tf

class NeuralNet(object):
    '''
    classdocs
    '''

    def __init__(self, name, model, trainer, sess):
        '''
        Constructor
        '''
        self.model = model
        self.trainer = trainer
        self.sess = sess
        self.train_step = trainer.train_step
        self.saver = tf.train.Saver(tf.global_variables(), scope=name, max_to_keep=None)

    def total_loss(self):
        return self.trainer.total_loss

    def forward(self, input_tensor):
        return self.model.forward(input_tensor)

    def save_model(self, tick):
        self.saver.save(self.sess, MODEL_SAVER_ID, global_step=tick)

    def restore_model(self, model_path, tick, verbose=False):
        ''' restore_model.

            Restore all the variables of the saved model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, MODEL_SAVER_ID + '-' + str(int(tick))))