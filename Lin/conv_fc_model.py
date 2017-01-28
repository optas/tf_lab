import tensorflow as tf
from global_variables import *
import tflearn
from tflearn.layers.conv import conv_2d,conv_2d_transpose
from tflearn.layers.core import fully_connected

def autoencoder(in_signal):
    layer = fully_connected(in_signal,nframe*imagelen*imagelen,weights_init='xavier')
    layer = tf.reshape(layer,[-1,imagelen,imagelen,nframe])
    layer = conv_2d(layer,24,4,2,activation='relu',weights_init='xavier')
    layer = conv_2d(layer,48,4,2,activation='relu',weights_init='xavier')
    layer = conv_2d(layer,96,4,2,activation='relu',weights_init='xavier')
    layer = conv_2d(layer,192,4,2,activation='relu',weights_init='xavier')
    layer = tf.reshape(layer,[-1,192*4*4])
    layer = fully_connected(layer,192*4*4,weights_init='xavier')
    layer = tf.reshape(layer,[-1,4,4,192])
    layer = conv_2d_transpose(layer,96,4,[8,8],strides=2,activation='relu',weights_init='xavier')
    layer = conv_2d_transpose(layer,48,4,[16,16],strides=2,activation='relu',weights_init='xavier')
    layer = conv_2d_transpose(layer,24,4,[32,32],strides=2,activation='relu',weights_init='xavier')
    layer = conv_2d_transpose(layer,nframe,4,[64,64],strides=2,activation='relu',weights_init='xavier')
    layer = tf.reshape(layer,[-1,nframe*imagelen*imagelen])
    layer = fully_connected(layer,Npoint*3)
    return layer


def loss(pred,groundtruth):
    loss = tf.reduce_mean(tf.pow(pred-groundtruth,2))
    tf.summary.scalar("loss",loss)
    return loss
