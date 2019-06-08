import tensorflow as tf
from tf_nndistance import nn_distance
import numpy as np

x = tf.placeholder(tf.float32, [1, 2, 3])
y = tf.placeholder(tf.float32, [1, 2, 3])

x_val = np.array([[[1,0,0],[0,1,0]]], dtype=np.float32)
y_val = np.array([[[0,0,4],[0,3,0]]], dtype=np.float32)

d1, _, d2, _ = nn_distance(x, y)

with tf.Session() as sess:
    d1_val, d2_val = sess.run([d1, d2], feed_dict={x: x_val, y: y_val})
    print(x_val)
    print(y_val)
    print(d1_val)
    print(d2_val)
