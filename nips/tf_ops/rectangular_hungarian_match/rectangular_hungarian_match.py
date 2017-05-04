import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
rectangular_hungarian_match_module = tf.load_op_library(os.path.join(BASE_DIR, 'rectangular_hungarian_match_so.so'))


def rectangular_hungarian_match(x):
  """ **Max** cost solution for assignment problem in O(n^3).

  Args
    x: BxNxM cost matrix for assignment problem. N <= M.

  Returns
    match: BxNxM match[i][j] = 1 if i is assigned to j, 0 elsewise
  """
  return rectangular_hungarian_match_module.rectangular_hungarian_match(x) # TODO: not sure which function
@tf.RegisterShape('RectangularHungarianMatch')
def _rectangular_hungarian_match_shape(op):
  shape1 = op.inputs[0].get_shape().with_rank(3)
  return [tf.TensorShape([shape1.dims[0], shape1.dims[1], shape1.dims[2]])]

if __name__=='__main__':
  with tf.Graph().as_default():
    with tf.device('/cpu:'+str(0)):
      x = tf.Variable(np.random.randint(0,100,size=(16,3,4)).astype('float32'))
      match = rectangular_hungarian_match(x)
      print match
    with tf.Session('') as sess:
      sess.run(tf.initialize_all_variables())
      x_val, match_val = sess.run([x, match])
      #print x_val, match_val
      print x_val[0,...].squeeze()
      print match_val[0,...].squeeze()
      """ Prints out (which is verified as correct):
      """
