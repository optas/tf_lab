import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from points_build_assignment import points_build_assignment


def _product(t):
  if isinstance(t, int):
    return t
  else:
    y = 1
    for x in t:
      y *= x
    return y

def _extra_feeds(extra_feed_dict, new_feeds):
  if not extra_feed_dict:
    return new_feeds
  r = {}
  r.update(extra_feed_dict)
  r.update(new_feeds)
  return r


def compute_numeric_jacobian(x, x_shape, x_data, y, y_shape, image_label, image_label_value, delta=0.001,
                              extra_feed_dict=None):
  """Computes the numeric Jacobian for dy/dx.

  Computes the numeric Jacobian by slightly perturbing the inputs and
  measuring the differences on the output.

  Args:
    x: the tensor "x".
    x_shape: the dimensions of x as a tuple or an array of ints.
    x_data: a numpy array as the input data for x
    y: the tensor "y".
    y_shape: the dimensions of y as a tuple or an array of ints.
    delta: the amount of perturbation we give to the input
    extra_feed_dict: dict that allows fixing specified tensor values
      during the jacobian calculation.

  Returns:
    A 2-d numpy array representing the Jacobian for dy/dx. It has "x_size" rows
    and "y_size" columns where "x_size" is the number of elements in x and
    "y_size" is the number of elements in y.
  """

  # To compute the jacobian, we treat x and y as one-dimensional vectors
  x_size = _product(x_shape) * (2 if x.dtype.is_complex else 1)
  y_size = _product(y_shape) * (2 if y.dtype.is_complex else 1)
  x_dtype = x.dtype.real_dtype.as_numpy_dtype
  y_dtype = y.dtype.real_dtype.as_numpy_dtype

  # Make sure we have the right types
  x_data = np.asarray(x_data, dtype=x.dtype.as_numpy_dtype)
  scale = np.asarray(2 * delta, dtype=y_dtype)[()]

  jacobian = np.zeros((x_size, y_size), dtype=x_dtype)

  # For each of the entry of x, we slightly perturbs this by adding and
  # subtracting a delta and then compute difference between the outputs. This
  # will give us one row of the Jacobian matrix.
  for row in range(x_size):
    x_pos = x_data.copy()
    x_neg = x_data.copy()
    x_pos.ravel().view(x_dtype)[row] += delta
    new_y = points_build_assignment(x, image_label)
    with tf.Session() as sess:
      y_pos = sess.run([new_y], feed_dict={x:x_pos, image_label:image_label_value})
    #y_pos = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_pos, image_label:image_label_value}))
    #y_pos = new_y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_pos, image_label:image_label_value}))
    print x, image_label, new_y
    print 'x_pos:', x_pos
    print 'image_label_value:', image_label_value
    print 'y_pos:', y_pos
    x_neg.ravel().view(x_dtype)[row] -= delta
    y_neg = y.eval(feed_dict=_extra_feeds(extra_feed_dict, {x: x_neg, image_label:image_label_value}))
    print 'x_neg:', x_neg
    print 'image_label_value:', image_label_value
    print 'y_neg:', y_neg
    diff = (y_pos - y_neg) / scale
    print 'scale:', scale
    print 'y_pos - y_neg:', y_pos-y_neg
    print 'diff:', diff
    raw_input()
    jacobian[row, :] = diff.ravel().view(y_dtype)

  return jacobian




class PointsBuildAssignmentTest(tf.test.TestCase):
  def test(self):
    with tf.Session() as sess:
      x_array = np.zeros((16,2,3))
      for i in range(2):
          for k in range(3):
            x_array[0,i,k] = (i+1)*(k+1);
      x = tf.constant(x_array, tf.float32) 

      y_array = np.zeros((16,2))
      y_array[0,...] = np.array([0,1])
      y = tf.constant(y_array, tf.int32)

      cost_matrix = points_build_assignment(x, y)
      cost_matrix_val = sess.run([cost_matrix], feed_dict={x:x_array, y:y_array})
      cost_matrix_val = cost_matrix_val[0]
      print x_array[0,...]
      print y_array[0,...]
      print cost_matrix_val[0,...]
      self.assertLess(np.sum(np.abs(cost_matrix_val[0,:,:] - [[1,2,0],[2,4,0],[3,6,0]])), 1e-4)

  def test_grad(self):
    with self.test_session():
      x = tf.constant(np.random.rand(16,2,3), dtype=tf.float32)
      y = tf.constant(np.random.randint(low=0, high=3, size=(16,2)), dtype=tf.int32)

      cost_matrix = points_build_assignment(x, y)
      err = tf.test.compute_gradient_error(x, (16,2,3), cost_matrix, (16,3,3))
      print 'error:', err
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main()
