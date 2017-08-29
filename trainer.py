'''
Created on Aug 28, 2017

@author: optas
'''
import tensorflow as tf
import os.path as osp

class Trainer(object):
    '''    
    '''

    def __init__(self, graph, optimizer, train_dir):
        '''
        Constructor
        '''
        self.graph
        self.train_dir = train_dir
        self.optimizer = optimizer
        self.train_writer = tf.summary.FileWriter(osp.join(train_dir, 'summaries'), self.graph)

 