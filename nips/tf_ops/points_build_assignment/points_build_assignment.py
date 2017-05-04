import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
points_build_assignment_module = tf.load_op_library(os.path.join(BASE_DIR, 'points_build_assignment_so.so'))

def points_build_assignment(x, y):
  """
  Args
    x: BxMxN float tensor of point log probabilities
    y: BxM int tensor of point class labels

  Returns
    cost_matrix: BxNxN float tensor
  """
  return points_build_assignment_module.points_build_assignment(x, y)
@tf.RegisterShape('PointsBuildAssignment')
def _points_build_assignment_shape(op):
  shape1 = op.inputs[0].get_shape().with_rank(3)
  shape2 = op.inputs[1].get_shape().with_rank(2)
  return [tf.TensorShape([shape1.dims[0], shape1.dims[2], shape1.dims[2]])]

def my_func(x, y, grad_cost_matrix): # grad_cost_matrix is the grad of the output
  b,m,n = x.shape
  grad = np.zeros_like(x).astype('float32')
  for bidx in range(b):
    for i in range(m):
        for k in range(n):
          grad[bidx,i,k] = grad_cost_matrix[bidx, k, y[bidx,i]]
  return grad

@tf.RegisterGradient('PointsBuildAssignment')
def _points_build_assignment_grad(op, grad_cost_matrix): # grad is BxNxN
  x = op.inputs[0] # BxMxN
  y = op.inputs[1] # BxM
  # {grad_x}_i_k) = {grad_cost_matrix}_k_yi 
  return [tf.py_func(my_func, [x,y,grad_cost_matrix], tf.float32), None]


if __name__=='__main__':
  with tf.Graph().as_default():
    with tf.device('/cpu:'+str(0)):
      x = tf.Variable(np.random.randn(16,2,3).astype('float32'))
      y = tf.Variable(np.random.randint(low=0, high=3, size=(16,2)).astype('int32'), dtype=tf.int32)
      cost_matrix = points_build_assignment(x, y)
      print cost_matrix
    with tf.Session('') as sess:
      sess.run(tf.initialize_all_variables())
      x_val, y_val, cost_matrix_val = sess.run([x, y, cost_matrix])
      print x_val[0,...].squeeze()
      print y_val[0,...].squeeze()
      print cost_matrix_val[0,...].squeeze()
