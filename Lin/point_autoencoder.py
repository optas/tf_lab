import tensorflow as tf
import numpy as np
import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from global_variables import *

# clean the logs_path
if os.path.exists(logs_path):
   shutil.rmtree(logs_path)
os.mkdir(logs_path)

def read_pointcloud(point_file):
    point_list = np.zeros([Npoint,3])
    line_count = 0
    for line in open(point_file):
        xyz = map(float,line.strip().split())
        for k in xrange(3):
            point_list[line_count][k] = xyz[k]
        line_count = line_count + 1
    return point_list

def read_data_sets(file_name):
    modelid = [line.strip().split('_pts.txt')[0] for line in os.listdir(file_name)]
    num_pc = len(modelid[0:100])
    point_data = np.zeros([num_pc,Npoint,3])
    for i in xrange(num_pc):
        point_file = os.path.join(file_name,modelid[i]+'_pts.txt')
        point_data[i] = read_pointcloud(point_file)
    return point_data

point_data = read_data_sets(chair_file_list)


weights = {
        'encoder_1_fc':tf.Variable(tf.random_normal([Npoint*3,imagelen*imagelen*nframe])),
        'encoder_2_conv':tf.Variable(tf.random_normal([4,4,nframe,24])),
        'encoder_3_conv':tf.Variable(tf.random_normal([4,4,24,48])),
        'encoder_4_conv':tf.Variable(tf.random_normal([4,4,48,96])),
        'encoder_5_fc':tf.Variable(tf.random_normal([96*4*4,256])),
        'decoder_1_fc':tf.Variable(tf.random_normal([256,96*4*4])),
        'decoder_2_deconv':tf.Variable(tf.random_normal([4,4,48,96])),
        'decoder_3_deconv':tf.Variable(tf.random_normal([4,4,24,48])),
        'decoder_4_deconv':tf.Variable(tf.random_normal([4,4,nframe,24])),
        'decoder_5_fc':tf.Variable(tf.random_normal([nframe*imagelen*imagelen,Npoint*3]))
}

biases = {
        'encoder_1_fc':tf.Variable(tf.random_normal([imagelen*imagelen*nframe])),
        'encoder_2_conv':tf.Variable(tf.random_normal([24])),
        'encoder_3_conv':tf.Variable(tf.random_normal([48])),
        'encoder_4_conv':tf.Variable(tf.random_normal([96])),
        'encoder_5_fc':tf.Variable(tf.random_normal([256])),
        'decoder_1_fc':tf.Variable(tf.random_normal([96*4*4])),
        'decoder_2_deconv':tf.Variable(tf.random_normal([48])),
        'decoder_3_deconv':tf.Variable(tf.random_normal([24])),
        'decoder_4_deconv':tf.Variable(tf.random_normal([nframe])),
        'decoder_5_fc':tf.Variable(tf.random_normal([Npoint*3]))
}

def conv2d(x,W,b,strides=1):
    # Conv2D wraper, with bias and relu activation
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def deconv2d(x, W, b,strides=1):
    x_shape = tf.shape(x)
    W_shape = tf.shape(W)
    out_shape = tf.pack([x_shape[0],x_shape[1]*2,x_shape[2]*2,W_shape[2]])
    x = tf.nn.conv2d_transpose(x,W,out_shape,[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def encoder(x):
    x = tf.reshape(x,[-1,Npoint*3])
    layer_1_fc = tf.add(tf.matmul(x,weights['encoder_1_fc']),biases['encoder_1_fc'])
    layer_1_relu = tf.nn.relu(layer_1_fc)

    layer_2_rs = tf.reshape(layer_1_relu,[-1,imagelen,imagelen,nframe])
    layer_2_conv = conv2d(layer_2_rs,weights['encoder_2_conv'],biases['encoder_2_conv'],strides=2)

    layer_3_conv = conv2d(layer_2_conv,weights['encoder_3_conv'],biases['encoder_3_conv'],strides=2)

    layer_4_conv = conv2d(layer_3_conv,weights['encoder_4_conv'],biases['encoder_4_conv'],strides=2)

    layer_5_rs = tf.reshape(layer_4_conv,[-1,4*4*96])
    layer_5_fc = tf.add(tf.matmul(layer_5_rs,weights['encoder_5_fc']),biases['encoder_5_fc'])
    return layer_5_fc

def decoder(x):
    layer_1_fc = tf.add(tf.matmul(x,weights['decoder_1_fc']),biases['decoder_1_fc'])
    layer_1_relu = tf.nn.relu(layer_1_fc)

    layer_2_rs = tf.reshape(layer_1_relu,[-1,4,4,96])
    layer_2_deconv = deconv2d(layer_2_rs,weights['decoder_2_deconv'],biases['decoder_2_deconv'],2)

    layer_3_deconv = deconv2d(layer_2_deconv,weights['decoder_3_deconv'],biases['decoder_3_deconv'],2)

    layer_4_deconv = deconv2d(layer_3_deconv,weights['decoder_4_deconv'],biases['decoder_4_deconv'],2)

    layer_5_rs = tf.reshape(layer_4_deconv,[-1,imagelen*imagelen*nframe])
    layer_5_fc = tf.add(tf.matmul(layer_5_rs,weights['decoder_5_fc']),biases['decoder_5_fc'])

    layer_6_rs = tf.reshape(layer_5_fc,[-1,Npoint,3])
    layer_6_tanh = tf.nn.tanh(layer_6_rs)
    return layer_6_tanh

# tf Graph Input
x = tf.placeholder(tf.float32,[None,Npoint,3],name='input_data')

encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

# prediction
y_pred = decoder_op
y_true = x

#Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred,2))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

#Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("cost",cost)

# Create summaries to visualize weights
#for var in tf.trainable_variables():
#    tf.summary.histogram(var.name,var)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        training_batch = zip(range(0, point_data.shape[0], batch_size),range(batch_size,point_data.shape[0]+1, batch_size))
        total_batch = len(training_batch)
        avg_cost = 0.0

        for i in xrange(total_batch):
            start,end = training_batch[i]
            batch_x = point_data[start:end]
            print str(start) + ' ' + str(end)
            _,c,summary = sess.run([optimizer,cost,merged_summary_op],feed_dict={x:batch_x})
            summary_writer.add_summary(summary,epoch * total_batch + i)
            avg_cost += c / total_batch * batch_size
        if epoch % display_step == 0:
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))

    print("Optimization done!")
