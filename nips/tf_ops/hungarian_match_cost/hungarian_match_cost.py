import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
hungarian_match_cost_module = tf.load_op_library(os.path.join(BASE_DIR, 'hungarian_match_cost_so.so'))


def hungarian_match_cost(cost_matrix, match):
  """
  Args
    cost_matrix: BxNxN float tensor
    match: BxNxN float tensor

  Returns
    cost: B float tensor
  """
  return hungarian_match_cost_module.hungarian_match_cost(cost_matrix, match)
@tf.RegisterShape('HungarianMatchCost')
def _hungarian_match_cost_shape(op):
  shape1 = op.inputs[0].get_shape().with_rank(3)
  shape2 = op.inputs[1].get_shape().with_rank(3)
  return [tf.TensorShape([shape1.dims[0]])]
@tf.RegisterGradient('HungarianMatchCost')
def _hungarian_match_cost_grad(op, grad_cost): # grad is B
  # grad(Xij) = grad(cost) * Mij
  match = op.inputs[1] # BxNxN
  return [tf.multiply(match, tf.expand_dims(tf.expand_dims(grad_cost,1),2)), None]

if __name__=='__main__':
  # TODO: write test code
  with tf.Graph().as_default():
    with tf.device('/cpu:'+str(0)):
      cost_matrix = tf.Variable(np.random.randn(16,3,3).astype('float32'))
      temp = np.array([[0,1,0],[0,0,1],[1,0,0]])
      match_array = np.zeros((16,3,3))
      for i in range(16):
        match_array[i,:,:] = temp
      match = tf.Variable(match_array.astype('float32'))
      cost = hungarian_match_cost(cost_matrix, match)
      print cost
    with tf.Session('') as sess:
      sess.run(tf.initialize_all_variables())
      cost_matrix_val, match_val, cost_val = sess.run([cost_matrix, match, cost])
      print cost_matrix_val[0,...].squeeze()
      print match_val[0,...].squeeze()
      print cost_val[0]
