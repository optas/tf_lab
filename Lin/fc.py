import tensorflow as tf
import numpy as np
import os
import sys

# append the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from data import *

tf.reset_default_graph()

#config
batch_size = 24
learning_rate = 0.01
training_epochs = 1000
display_step = 5
logs_path="./tmp/logs/"
saver_path = './tmp/'

def init_weights(shape):
    with tf.device("/gpu:0"):
        v = tf.Variable(tf.random_normal(shape,stddev=0.01))
        return v

def model(X,w_1,w_2,w_3,w_4,w_5):
    X_ = tf.reshape(X,[-1,Npoint*3])
    h1 = tf.nn.relu(tf.matmul(X_,w_1))
    h2 = tf.nn.relu(tf.matmul(h1,w_2))
    h3 = tf.nn.relu(tf.matmul(h2,w_3))
    h4 = tf.nn.relu(tf.matmul(h3,w_4))
    h5 = tf.nn.tanh(tf.matmul(h4,w_5))
    out = tf.reshape(h5,[-1,Npoint,3])
    return out

X = tf.placeholder("float",[None, Npoint, 3])
Y = tf.placeholder("float",[None, Npoint, 3])

w_1 = init_weights([Npoint*3,Npoint*4])
w_2 = init_weights([Npoint*4,Npoint])
w_3 = init_weights([Npoint,int(Npoint*0.2)])
w_4 = init_weights([int(Npoint*0.2),Npoint])
w_5 = init_weights([Npoint,Npoint*3])

# Construct a linear model
with tf.name_scope('Model'):
    pred = model(X,w_1,w_2,w_3,w_4,w_5)

# Mean Sequared Error
with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.pow(pred-Y,2)/batch_size)

#Initaliazing the variables
init = tf.global_variables_initializer()

#Create a summary to monitor cost tensor
tf.summary.scalar("cost",cost)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # you  need to initalize all variables
    sess.run(init)

    writer = tf.train.SummaryWriter('./tmp/logs',graph=tf.get_default_graph())
    loadData()

    for epoch in range(100):
        training_batch = zip(range(0,len(trX),batch_size),range(batch_size,len(trX)+1,batch_size))
        avg_cost = 0
        total_batch = len(training_batch)
        save_path = saver.save(sess,os.path.join(saver_path,"model_" + str(epoch)))
        print("Model saved in file: %s" % save_path)

        for i in range(total_batch):
            start,end = training_batch[i]
            _,c,summary = sess.run([optimizer,cost,summary_op],feed_dict={X:trX[start:end],Y:trY[start:end]})
            writer.add_summary(summary,i + epoch * total_batch)
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))

