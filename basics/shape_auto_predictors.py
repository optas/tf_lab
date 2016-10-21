'''
Created on October 11, 2016

@author: optas
'''

import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn_cell

flags = tf.flags
FLAGS = flags.FLAGS
FLAGS.use_fp16 = False

def data_type():
        return tf.float16 if FLAGS.use_fp16 else tf.float32

class SAP_Input(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#         self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)

    
class Congif(object):
    
    
class Shape_Auto_Predictor_Toy(object):
    """The Shape Predictor Toy model."""

    def __init__(self, is_training, config, input_):
        self._input = input_
        
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        forget_bias = config.forget_bias
        inputs = input_.input_data
                
                                
        encoder = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=True)
        
        if is_training and config.keep_prob < 1:
            encoder = rnn_cell.DropoutWrapper(encoder, output_keep_prob=config.keep_prob)
        
        self._initial_state = encoder.zero_state(batch_size, data_type())    
        
        with tf.device("/cpu:0"):
            # Prepare data shape to match `rnn` function requirements            
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
            inputs = input_.input_data            
            inputs = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, num_steps, inputs)] 
        
        outputs, state = tf.nn.rnn(encoder, inputs, initial_state=self._initial_state)
        
        
        output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return
        
        