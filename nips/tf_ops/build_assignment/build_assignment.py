import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
build_assignment_module = tf.load_op_library(os.path.join(BASE_DIR, 'build_assignment_so.so'))

def build_assignment(x, y):
  """
  Args
    x: BxHxWxN float tensor of image pixel log probabilities
    y: BxHxW int tensor of image pixel class labels

  Returns
    cost_matrix: BxNxN float tensor
  """
  return build_assignment_module.build_assignment(x, y)
@tf.RegisterShape('BuildAssignment')
def _build_assignment_shape(op):
  shape1 = op.inputs[0].get_shape().with_rank(4)
  shape2 = op.inputs[1].get_shape().with_rank(3)
  return [tf.TensorShape([shape1.dims[0], shape1.dims[3], shape1.dims[3]])]

def my_func(x, y, grad_cost_matrix): # grad_cost_matrix is the grad of the output
  b,h,w,n = x.shape
  grad = np.zeros_like(x).astype('float32')
  for bidx in range(b):
    for i in range(h):
      for j in range(w):
        for k in range(n):
          grad[bidx,i,j,k] = grad_cost_matrix[bidx, k, y[bidx,i,j]]
  return grad

@tf.RegisterGradient('BuildAssignment')
def _build_assignment_grad(op, grad_cost_matrix): # grad is BxNxN
  x = op.inputs[0] # BxHxWxN
  y = op.inputs[1] # BxHxW
  # {grad_x}_i_j_k) = {grad_cost_matrix}_k_yij 
  return [tf.py_func(my_func, [x,y,grad_cost_matrix], tf.float32), None]


if __name__=='__main__':
  with tf.Graph().as_default():
    with tf.device('/cpu:'+str(0)):
      #x = tf.ones((3,5,5,2))
      #y = tf.zeros((3,5,5), dtype=tf.int32)
      x = tf.Variable(np.random.randn(16,2,2,3).astype('float32'))
      y = tf.Variable(np.random.randint(low=0, high=3, size=(16,2,2)).astype('int32'), dtype=tf.int32)
      cost_matrix = build_assignment(x, y)
      print cost_matrix
    with tf.Session('') as sess:
      sess.run(tf.initialize_all_variables())
      x_val, y_val, cost_matrix_val = sess.run([x, y, cost_matrix])
      print x_val[0,...].squeeze()
      print y_val[0,...].squeeze()
      print cost_matrix_val[0,...].squeeze()
      """ Prints out (which is verified as correct):
[[[ 0.47740752  0.29287109 -0.76430809]
  [-0.43345165 -0.17250288 -0.55713695]]

 [[-0.55650556 -0.14719154 -0.28433409]
  [-0.36214039 -0.14317064  0.18925352]]]

[[2 0]
 [2 1]]

[[-0.43345165 -0.36214039 -0.07909805]
 [-0.17250288 -0.14317064  0.14567955]
 [-0.55713695  0.18925352 -1.04864216]]
      """
