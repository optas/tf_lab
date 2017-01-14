import tensorflow as tf
import numpy as np
import os
from global_variables import *
import tensorflow.contrib.slim as slim


trX = []
trY = []

def readPointFile(filename):
    pointlist = np.zeros([Npoint,3])
    numpoint = 0
    for l in open(filename):
        if not l.startswith('#'):
            x,y,z= map(float,l.strip().split())
            pointlist[numpoint][0] = x
            pointlist[numpoint][1] = y
            pointlist[numpoint][2] = z
            numpoint = numpoint + 1
    return pointlist

def load_data():
    model_list = [line.strip().split('_pts.txt')[0] for line in os.listdir(filelist)]
    for pc_file in model_list:
        filename = os.path.join(filelist,pc_file+'_pts.txt')
        pointcloud = readPointFile(filename)
        trX.append(pointcloud)
        trY.append(pointcloud)


n_hidden = 4
hidden_layer_sizes = [Npoint*4, Npoint, int(0.2*Npoint), Npoint, Npoint*3]

def autoendoder(in_signal):
    in_signal = tf.reshape(in_signal, [-1, Npoint*3])
    layer = slim.fully_connected(in_signal, hidden_layer_sizes[0], activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[1],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[2],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[3],activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.fully_connected(layer, hidden_layer_sizes[4],activation_fn=None)
    layer = tf.tanh(layer)
    return tf.reshape(layer, [-1, Npoint, 3])

X = tf.placeholder("float",[None, Npoint, 3])
Y = tf.placeholder("float",[None, Npoint, 3])

# Construct a linear model
pred = autoendoder(X)

# Mean Sequared Error
cost = tf.reduce_mean(tf.pow(pred-Y,2))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    # you  need to initalize all variables
    tf.initialize_all_variables().run()

    load_data()
    for i in range(1000):
        training_batch = zip(range(0,len(trX),batch_size),range(batch_size,len(trX)+1,batch_size))
        for start,end in training_batch:
            sess.run(optimizer,feed_dict={X:trX[start:end],Y:trY[start:end]})
