import tensorflow as tf
import numpy as np
from hungarian_match_cost import hungarian_match_cost

class HungarianMatchCostTest(tf.test.TestCase):
  def test(self):
    with tf.Session() as sess:
      cost_matrix = tf.constant(np.array([[[5,10.2,4.2],[3.7,8,9],[19,5,7]]]), dtype=tf.float32)
      match = tf.constant(np.array([[[0,1,0],[0,0,1],[1,0,0]]]), dtype=tf.float32)
      cost = hungarian_match_cost(cost_matrix, match)
      cost_val = sess.run([cost])
      self.assertLess(cost_val[0][0]-38.2, 1e-4)

  def test_grad(self):
    with self.test_session():
      cost_matrix = tf.constant(np.array([[[5,10.2,4.2],[3.7,8,9],[19,5,7]]]), dtype=tf.float32)
      print cost_matrix
      match = tf.constant(np.array([[[0,1,0],[0,0,1],[1,0,0]]]), dtype=tf.float32)
      print match
      cost = hungarian_match_cost(cost_matrix, match)
      err = tf.test.compute_gradient_error(cost_matrix, (1,3,3), cost, (1,))
      print err
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
