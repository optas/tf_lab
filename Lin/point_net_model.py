import tensorflow as tf
from global_variables import *
import tflearn
from tflearn.layers.conv import conv_1d,conv_2d,conv_3d,conv_3d_transpose
from tflearn.layers.core import fully_connected
import tensorflow.contrib.slim as slim

def autoencoder(in_signal):
    layer = tf.reshape(in_signal,[-1,Npoint,3])
    layer = conv_1d(layer,64,1,1,activation='relu')
    layer = conv_1d(layer,64,1,1,activation='relu')
    layer = conv_1d(layer,64,1,1,activation='relu')
    layer = conv_1d(layer,128,1,1,activation='relu')
    layer = conv_1d(layer,1024,1,1,activation='relu')
    layer = tf.reduce_max(layer,1)
    layer = fully_connected(layer,1024,activation='relu')
    layer = fully_connected(layer,32*4*4*4,activation='relu')
    layer = tf.reshape(layer,[-1,3,4,4,4])
    layer = conv_2d_transpose(layer,12,4,[8,8,8],strides=2,activation='relu',weights_init='xavier')
    #layer = conv_2d_transpose(layer,48,4,[16,16],strides=2,activation='relu',weights_init='xavier')
    #layer = conv_2d_transpose(layer,24,4,[32,32],strides=2,activation='relu',weights_init='xavier')
    #layer = conv_2d_transpose(layer,nframe,4,[64,64],strides=2,activation='relu',weights_init='xavier')
    #layer = tf.reshape(layer,[-1,nframe*imagelen*imagelen])
    #layer = fully_connected(layer,Npoint*3)
    return layer


def loss(pred,groundtruth):
    loss = tf.reduce_mean(tf.pow(pred-groundtruth,2))
    tf.summary.scalar("loss",loss)
    return loss
