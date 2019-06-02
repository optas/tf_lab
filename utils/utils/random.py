"""
Created on Jun 2, 2019

@author: optas
"""

import tensorflow as tf


def randomly_mix_tensors(x, y):    
    """ Mixes the contents of x and y, batch-wise.
        Args:
            x, y (tf.Tensor): B x D1 x ... x DM, two tensors of the same size to be mixed.
        Returns:
            merged_shuffled (tf.Tensor): B x 2 x D1 x ... x DM
            first_col: (tf.Tensor): B, binary indicator: if the first object of 
            ``` merged_shuffled ``` belongs to x (if not belongs to y).
        
        Example: 
            x = [x0, x1, x2], y = [y0, y1, y2]
            A, B = randomly_mix_tensors(x, y)
            B = [0, 1, 1] and A =[x0, y0; y1, x1; y2, x2]   
    """
    b_size = tf.shape(x)[0]
    dummy = tf.shape(y)[0]    
    with tf.control_dependencies([tf.assert_equal(b_size, dummy)]):
        first_col = tf.random_uniform(shape=[b_size], maxval=2, dtype=tf.int32)
        batch_idx = tf.range(b_size)
        second_col = 1 - first_col
        merged = tf.stack([x, y], 1)
        column_1 = tf.gather_nd(merged, tf.stack([batch_idx, first_col], 1))
        column_2 = tf.gather_nd(merged, tf.stack([batch_idx, second_col], 1))
        merged_shuffled = tf.stack([column_1, column_2], 1)
    return merged_shuffled, first_col


