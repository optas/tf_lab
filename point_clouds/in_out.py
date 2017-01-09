import tensorflow as tf
import numpy as np
import os.path as osp
import glob

from autopredictors.scripts.helper import points_extension
from geo_tool.in_out.soup import load_crude_point_cloud


def load_crude_point_clouds(top_directory):
    pclouds = []
    model_names = []
    for file_name in glob.glob(osp.join(top_directory, '*' + points_extension)):
        pclouds.append(load_crude_point_cloud(file_name))
        model_name = osp.basename(file_name).split('_')[0]
        model_names.append(model_name)
    return pclouds, model_names


def load_filenames_of_input_data(top_directory):
    res = []
    for file_name in glob.glob(osp.join(top_directory, '*' + points_extension)):
        res.append(file_name)

    print '%d files containing  point clouds were found.' % (len(res), )
    return tf.convert_to_tensor(res, dtype=tf.string)


def in_out_placeholders(configuration):
    n = configuration.n_points
    e = configuration.original_embedding
    b = configuration.batch_size
    in_signal = tf.placeholder(dtype=tf.float32, shape=(b, n, e), name='input_pclouds')
    gt_signal = tf.placeholder(dtype=tf.float32, shape=(b, n, e), name='output_pclouds')
    return in_signal, gt_signal


# batch_size = 24
# Npoint = 2700
# learning_rate = 0.01
# training_epochs = 1000
# trX = []
# trY = []
# n_hidden = 4
# hidden_layer_sizes = [Npoint*4, Npoint, int(0.2*Npoint), Npoint, Npoint*3]


# segs_extension = '_segs.txt'
# feat_extension = '_feat.txt' 



# 
# 
# if __name__ == '__main__':        
#     # Construct a linear model
#     pred = autoendoder(X)
#     
#     # Mean Squared Error
#     cost = tf.reduce_mean(tf.pow(pred-Y,2))
#     
#     # Gradient descent
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#     
#     
#     with tf.Session() as sess:
#         # you  need to initialize all variables
#         tf.initialize_all_variables().run()
#     
#         load_data()
#         for i in range(1000):
#             training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
#             for start,end in training_batch:
#                 sess.run(optimizer,feed_dict={X:trX[start:end],Y:trY[start:end]})
