'''
Created on February 2, 2017
 
@author: optas
'''
 
import tensorflow as tf
from tflearn import initializations
from . import utils
from . import nn
 
 
def conv_3d_transpose(incoming, nb_filter, filter_size, output_shape,
                      strides=1, padding='same',
                      bias=True, weights_init='uniform_scaling',
                      bias_init='zeros', regularizer=None, weight_decay=0.001,
                      trainable=True, restore=True, reuse=False, scope=None,
                      name="Conv3DTranspose"):
 
    """ Convolution 3D Transpose.
    This operation is sometimes called "deconvolution" after (Deconvolutional
    Networks)[http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf], but is
    actually the transpose (gradient) of `conv_3d` rather than an actual
    deconvolution.
    Input:
        5-D Tensor [batch, depth, height, width, in_channels].
    Output:
        5-D Tensor [batch, new depth, new height, new width, nb_filter].
    Arguments:
        incoming: `Tensor`. Incoming 5-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        output_shape: `list of int`. Dimensions of the output tensor.
            Can optionally include the number of conv filters.
            [new depth, new height, new width, nb_filter] or [new depth, new height, new width].
        strides: `int` or list of `int`. Strides of conv operation.
            Default: [1 1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv2DTranspose'.
    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.
    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 5, "Incoming Tensor shape must be 5-D"
 
    filter_size = utils.autoformat_filter_conv3d(filter_size,
                                                 nb_filter,
                                                 input_shape[-1])
    strides = utils.autoformat_stride_3d(strides)
    padding = utils.autoformat_padding(padding)
 
    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)
 
    with vscope as scope:
        name = scope.name
 
        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
#         W_regul = None
#         if regularizer:
#             W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
#         W = vs.variable('W', shape=filter_size,
#                         regularizer=W_regul, initializer=W_init,
#                         trainable=trainable, restore=restore)
 
        wd = 0
        W = nn._variable_with_weight_decay('W', filter_size, W_init, wd, trainable=trainable)
 
#         Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)
 
#         b = None
#         if bias:
#             if isinstance(bias_init, str):
#                 bias_init = initializations.get(bias_init)()
 
#             b = vs.variable('b', shape=nb_filter, initializer=bias_init,
#                             trainable=trainable, restore=restore)
 
        b = nn._bias_variable(shape=nb_filter)
        # Track per layer variables
#             tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)
 
        # Determine the complete shape of the output tensor.
        batch_size = tf.gather(tf.shape(incoming), tf.constant([0]))
        if len(output_shape) == 3:
            output_shape = output_shape + [nb_filter]
        elif len(output_shape) != 4:
            raise Exception("output_shape length error: " + str(len(output_shape)) + ", only a length of 3 or 4 is supported.")
        complete_out_shape = tf.concat(0, [batch_size, tf.constant(output_shape)])
 
        inference = tf.nn.conv3d_transpose(incoming, W, complete_out_shape,
                                           strides, padding)
 
        # Reshape tensor so its shape is correct.
        inference.set_shape([None] + output_shape)
 
        if b: inference = tf.nn.bias_add(inference, b)
 
#         if isinstance(activation, str):
#             inference = activations.get(activation)(inference)
#         elif hasattr(activation, '__call__'):
#             inference = activation(inference)
#         else:
#             raise ValueError("Invalid Activation.")
 
        # Track activations.
#         tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)
 
    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b
 
    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)
 
    return inference
