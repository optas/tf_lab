'''
Created on Jan 9, 2017

@author: optas
'''

import tensorflow as tf
from . utils import format_scope_name

SUMMARIES_COLLECTION = 'tf_lab_summaries'  # Keeping all summaries in this collection, each summary being stored as part of a dictionary.
SUMMARY_TAG = 'tag'                        # Used as the key on a dictionary storing the name of the summary operation.
SUMMARY_TENSOR = 'tensor'                  # Used as the key on a dictionary storing the tensor of the summary operation.


def get_summary_if_exists(tag):
    """ summary_exists.
    Retrieve a summary exists if exists, or None.
    Arguments:
        tag: `str`. The summary name.
    """
    return next((item[SUMMARY_TENSOR] for item in tf.get_collection(SUMMARIES_COLLECTION) if item[SUMMARY_TAG] == tag), None)


def get_summary(stype, tag, value=None, collection_key=None, break_if_exists=False):
    """ get_summary.

    Create or retrieve a summary. It keep tracks of all graph summaries
    through summary_tags collection. If a summary tags already exists,
    it will return that summary tensor or raise an error (according to
    'break_if_exists').
    Arguments:
        stype: `str`. Summary type: 'histogram', 'scalar' or 'image'.
        tag: `str`. The summary tag (name).
        value: `Tensor`. The summary initialization value. Default: None.
        collection_key: `str`. If specified, the created summary will be
            added to that collection (optional).
        break_if_exists: `bool`. If True, if a summary with same tag already
            exists, it will raise an exception (instead of returning that
            existing summary).
    Returns:
        The summary `Tensor`.
    """
    summ = next((item for item in tf.get_collection(SUMMARIES_COLLECTION) if
                 item[SUMMARY_TAG] == tag), None)

    if not summ:
        if value is None:
            raise Exception("Summary doesn't exist, a value must be "
                            "specified to initialize it.")
        if stype == "histogram":
            summ = tf.summary.histogram(tag, value)
        elif stype == "scalar":
            summ = tf.summary.scalar(tag, value)
        elif stype == "image":
            summ = tf.summary.image(tag, value)
        else:
            raise ValueError("Unknown summary type: '" + str(stype) + "'")

        tf.add_to_collection(SUMMARIES_COLLECTION, {SUMMARY_TAG: tag, SUMMARY_TENSOR: summ})

        if collection_key:
            tf.add_to_collection(collection_key, summ)
    elif break_if_exists:
        raise ValueError("Error: Summary tag already exists! (to ignore this "
                         "error, set add_summary() parameter 'break_if_exists'"
                         " to False)")
    else:
        summ = summ[SUMMARY_TENSOR]

    return summ


def add_gradients_summary(grads, name_prefix="", name_suffix="", collection_key=None):
    """ add_gradients_summary.
    Add histogram summary for given gradients.
    Arguments:
        grads: A list of `Tensor`. The gradients to summarize.
        name_prefix: `str`. A prefix to add to summary scope.
        name_suffix: `str`. A suffix to add to summary scope.
        collection_key: `str`. A collection to store the summaries.
    Returns:
        The list of created gradient summaries.
    """

    # Add histograms for gradients.
    summ = []
    for grad, var in grads:
        if grad is not None:
            summ_name = format_scope_name(var.op.name, name_prefix,
                                          "Gradients/" + name_suffix)
            summ_exists = get_summary_if_exists(summ_name)
            if summ_exists is not None:
                tf.add_to_collection(collection_key, summ_exists)
                summ.append(summ_exists)
            else:
                summ.append(get_summary("histogram", summ_name, grad,
                                        collection_key))
    return summ


def _activation_summary(x):
    '''Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: (tf.Tensor)
    Returns:
        nothing
    '''

# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
# session. This helps the clarity of presentation on tensorboard.
# tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tensor_name = x.op.name
    tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
    tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def trainable_variables(in_graph=None):
    if in_graph is None:
        return tf.trainable_variables()
    else:
        return in_graph.get_collection('trainable_variables')


def count_trainable_parameters(in_graph=None):
    if in_graph is None:
        in_graph = tf.get_default_graph()

    total_parameters = 0
    # for variable in tf.trainable_variables():
    for variable in in_graph.get_collection('trainable_variables'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters


def hist_summary_of_trainable(in_graph=None):
    summaries = []
    with tf.device('/cpu:0'):
        for var in trainable_variables(in_graph):
#             summaries.append(tf.histogram_summary(var.op.name, var))
            summaries.append(tf.summary.histogram(var.op.name, var))
    return summaries


def sparsity_summary_of_trainable():
    summaries = []
    with tf.device('/cpu:0'):
        for var in tf.trainable_variables():
            summaries.append(tf.scalar_summary('sparsity_' + var.op.name, tf.nn.zero_fraction(var)))
    return summaries
