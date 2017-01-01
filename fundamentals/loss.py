'''
Created on December 31, 2016   
'''

import tensorflow as tf
import numpy as np

class Loss():
    
    @staticmethod
    def l2_loss(prediction, ground_truth):    
        ground_truth = tf.cast(ground_truth, tf.float32)
        loss = tf.square(prediction - ground_truth)
        return tf.reduce_mean(loss, name='l2_loss')
    
    @staticmethod
    def soft_max_loss(prediction, ground_truth, sparse_gt=True):
        '''
        If one-hot vectors are used for the ground_truth then the sparse_gt should be False.
        '''       
        if sparse_gt:
            ground_truth = tf.cast(ground_truth, tf.int64)
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, ground_truth, name='cross_entropy_per_example')
        else:
            labels = tf.cast(target, tf.float32)
            entropy = tf.nn.softmax_cross_entropy_with_logits(prediction, ground_truth, name='cross_entropy_per_example')

        return tf.reduce_mean(entropy, name='cross_entropy')
