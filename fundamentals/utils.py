'''
Created on Feb 2, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np
import os


def set_visible_GPUs(accessible=[0]):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # See issue #152 stack-overflow.
    accessible = '"' + ','.join(str(e) for e in accessible) + '"'
    os.environ["CUDA_VISIBLE_DEVICES"] = accessible


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def autoformat_padding(padding):
    if padding in ['same', 'SAME', 'valid', 'VALID']:
        return str.upper(padding)
    else:
        raise Exception("Unknown padding! Accepted values: 'same', 'valid'.")


# Auto format filter size
# Output shape: (rows, cols, input_depth, out_depth)
def autoformat_filter_conv3d(fsize, in_depth, out_depth):
    if isinstance(fsize, int):
        return [fsize, fsize, fsize, in_depth, out_depth]
    elif isinstance(fsize, (tuple, list)):
        if len(fsize) == 3:
            return [fsize[0], fsize[1], fsize[2], in_depth, out_depth]
        else:
            raise Exception("filter length error: " + str(len(fsize)) + ", only a length of 3 is supported.")
    else:
        raise Exception("filter format error: " + str(type(fsize)))


# Auto format stride for 3d convolution
def autoformat_stride_3d(strides):
    if isinstance(strides, int):
        return [1, strides, strides, strides, 1]
    elif isinstance(strides, (tuple, list)):
        if len(strides) == 3:
            return [1, strides[0], strides[1], strides[2], 1]
        elif len(strides) == 5:
            assert strides[0] == strides[4] == 1, "Must have strides[0] = strides[4] = 1"
            return [strides[0], strides[1], strides[2], strides[3], strides[4]]
        else:
            raise Exception("strides length error: " + str(len(strides)) + ", only a length of 3 or 5 is supported.")
    else:
        raise Exception("strides format error: " + str(type(strides)))


# Auto format kernel for 3d convolution
def autoformat_kernel_3d(kernel):
    if isinstance(kernel, int):
        return [1, kernel, kernel, kernel, 1]
    elif isinstance(kernel, (tuple, list)):
        if len(kernel) == 3:
            return [1, kernel[0], kernel[1], kernel[2], 1]
        elif len(kernel) == 5:
            assert kernel[0] == kernel[4] == 1, "Must have kernel_size[0] = kernel_size[4] = 1"
            return [kernel[0], kernel[1], kernel[2], kernel[3], kernel[4]]
        else:
            raise Exception("kernel length error: " + str(len(kernel)) + ", only a length of 3 or 5 is supported.")
    else:
        raise Exception("kernel format error: " + str(type(kernel)))
