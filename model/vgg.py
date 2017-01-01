'''
Created on October 7, 2016

'''

import tensorflow as tf
import numpy as np

def vgg_m_conv(in_signal, keep_prob, stddev = 5e-2):
    '''
    DOI: Return of the Devil in the Details: Delving Deep into Convolutional Nets: Ken Chatfield et al.    
    '''    
    conv = convolutional_layer
    
    layer = conv(in_signal, n_filters=96, filter_size=[7,7], stride=2, padding='SAME', stddev=stddev, name="conv_1")
    layer = max_pool(relu(layer), ksize=(3,3), stride=(2,2), name='max_pool_1')

    layer = conv(layer, n_filters=256, filter_size=[5,5], stride=2, padding='SAME', stddev=stddev, name="conv_2")
    layer = max_pool(relu(layer), ksize=(3,3), stride=(2,2), name='max_pool_2')
                        
    layer = conv(layer, n_filters=512, filter_size=[3,3], stride=1, padding='SAME', stddev=stddev, name="conv_3")
    layer = relu(layer)

    layer = conv(layer, n_filters=512, filter_size=[3,3], stride=1, padding='SAME', stddev=stddev, name="conv_4")
    layer = relu(layer)
    
    layer = conv(layer, n_filters=512, filter_size=[3,3], stride=1, padding='SAME', stddev=stddev, name="conv_5")
    layer = max_pool(relu(layer), ksize=(3,3), stride=(2,2), name='max_pool_3')
    
    layer = fully_connected_layer(layer, 4096, stddev=stddev, name="fc_6")
    layer = dropout(relu(layer), keep_prob)
    
    layer = fully_connected_layer(layer, 4096, stddev=stddev, name="fc_7")
    layer = dropout(relu(layer), keep_prob)

    return layer