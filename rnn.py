import tensorflow as tf


def deep_lstm(n_layers, n_hidden, out_keep_prob=1, input_keep_prob=1, activation=tf.nn.tanh):
    cells = []
    for _ in range(n_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, activation=activation)

        if out_keep_prob is not 1 or input_keep_prob is not 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=out_keep_prob, input_keep_prob=input_keep_prob)

        cells.append(cell)

    model = tf.nn.rnn_cell.MultiRNNCell(cells)
    return model


def length_of_sequence(sequence):
    '''
        Input: (Tensor) batch size x max length x features
        Returns: the length of each sequence in the batch.
        Precondition: Each sequence with smaller length that max length, is padded with zeros.
    '''
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def last_relevant_rnn_output(output, length):
    '''
    Returns for a batch of output tensors of a dynamic_rnn those corresponding
    to the last input frame. I.e., cleverly drop the outputs corresponding to the padded input.
    Notes:
    In numpy this would just be output[:, length - 1], but we need the indexing to be part of the compute graph.
    Works likes this: we flatten the output tensor to shape: frames in all examples x output size.
    Then we construct an index into that by creating a tensor with the start indices for each example
    tf.range(0, batch_size) * max_length and add the individual sequence lengths to it.
    tf.gather() then performs the actual indexing. 
    NOTE:
    The best approach in the future will be to use:
        last_rnn_output = tf.gather_nd(output, tf.pack([tf.range(batch_size), seqlen-1], axis=1))
    '''
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)


def get_state_reset_op(state_variables, cell, batch_size):
    # Return an operation to set each variable in a list of LSTMStateTuples to zero
    zero_states = cell.zero_state(batch_size, tf.float32)
    return get_state_update_op(state_variables, zero_states)



  # # # # NEW STUFF TO CONSIDER

# def lstm_w_attention_get_state_variables(batch_size, cell):
#     # For each layer, get the initial state and make a variable out of it
#     # to enable updating its value.
#     state_variables = []
#     for layer in cell.zero_state(batch_size, tf.float32):
#         for k in layer:
#             if type(k) == tf.contrib.rnn.LSTMStateTuple:
#                 state_c, state_h = k
#                 state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
#                     tf.Variable(state_c, trainable=False),tf.Variable(state_h, trainable=False)))
#     # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
#     return tuple(state_variables)

# def deep_lstm(n_layers, n_hidden, dropout_prob=None, attn_len=None, activation=tf.nn.tanh):
#     cells = []
#     for _ in range(n_layers):
#         
#         cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, activation=activation)
#         
#         if attn_len is not None:
#             cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_len)
#     
#         if dropout_prob is not None:            
#             cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0 - dropout_prob)
# #             cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
# #                                                  input_keep_prob=1.0 - dropout_prob,
# #                                                  output_keep_prob=1.0 - dropout_prob,
# #                                                  state_keep_prob = 1.0 - dropout_prob
# #                                                 )
#             
#         cells.append(cell)
#     model = tf.nn.rnn_cell.MultiRNNCell(cells)
#     return model