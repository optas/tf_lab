'''
Created on December 31, 2016
'''

import tensorflow as tf


class Loss():

    @staticmethod
    def l2_loss(prediction, ground_truth):
        '''L2 norm without the sqrt'''
        ground_truth = tf.cast(ground_truth, tf.float32)
        loss = tf.square(prediction - ground_truth)
        return tf.reduce_mean(loss, name='l2_loss')

    @staticmethod
    def cross_entropy_loss(u_logits, ground_truth, sparse_gt=True):
        '''
        Input:
        u_logits: unscaled logits.
        sparse_gt: True if ground_trush is comprised by one-hot vectors.
        '''
        if sparse_gt:
            ground_truth = tf.cast(ground_truth, tf.int64)
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u_logits, labels=ground_truth, name='cross_entropy_per_example')
        else:
            ground_truth = tf.cast(ground_truth, tf.float32)
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=u_logits, labels=ground_truth, name='cross_entropy_per_example')

        return tf.reduce_mean(entropy, name='cross_entropy')

    @staticmethod
    def cosine_distance_loss(prediction, ground_truth, epsilon=10e-6):
        cosine = tf.reduce_sum(prediction * ground_truth, 2)
        norm = tf.sqrt(tf.reduce_sum(prediction * prediction, 2) + epsilon)
        return tf.reduce_mean(- 1.0 * tf.abs(cosine / norm))
    