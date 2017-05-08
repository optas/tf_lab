'''
    Input Nx3 point cloud (normalized to unit sphere and zero mean), Output 40 probabilities for ModelNet40 classes
    Uses PointNet vanilla model with accuracy around 87.2%
'''
import tensorflow as tf
import numpy as np
import os
import sys

from general_tools.simpletons import iterate_in_chunks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pointnet_cls_basic as model

# from .. data_sets.model_net import net_40_classes   TODO FIX THIS


NUM_CLASSES = 40

# net_40_classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
#                   'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door',
#                   'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
#                   'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano',
#                   'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
#                   'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

# classes_to_integers

def softmax(x):
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def get_model(model_path, gpu_index, batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_index)):
            pointclouds_pl, _ = model.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            logits, _ = model.get_model(pointclouds_pl, is_training_pl)
            print logits
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, model_path)
        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'logits': logits}
        return sess, ops


def inference(sess, ops, pc, batch_size, use_softmax=True):
    ''' pc: BxNx3 array, return BxN probs/logits '''
    assert pc.shape[0] % batch_size == 0
    num_batches = pc.shape[0] / batch_size
    logits = np.zeros((pc.shape[0], NUM_CLASSES))
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
                     ops['is_training_pl']: False}
        batch_logits = sess.run(ops['logits'], feed_dict=feed_dict)
        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
    if use_softmax:
        return softmax(logits)
    else:
        return logits


if __name__ == '__main__':

    pclouds = np.load(sys.argv[1])
    gpu_index = int(sys.argv[2])
    class_name = sys.argv[3]

    pclouds = pclouds[pclouds.keys()[0]]

    # The networks was trained with zero-mean pclouds AND in UNIT sphere. Thus we apply this transformation here.
    pclouds = pclouds - np.expand_dims(np.mean(pclouds, axis=1), 1)
    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    dist = np.expand_dims(np.expand_dims(dist, 1), 2)
    pclouds = pclouds / dist

    # We also swap the axis to align the models with the model-net ones.
#     pclouds_rot = np.empty_like(pclouds)
    for i, pc in enumerate(pclouds):
        pclouds[i] = pc[:, [0, 2, 1]]

    if class_name == 'chair':
        class_index = 8
    elif class_name == 'car':
        class_index = 7
    else:
        assert(False)

    batch_size = 100

    sess, ops = get_model(model_path='log_to_panos/model.ckpt', gpu_index=gpu_index, batch_size=batch_size, num_point=pclouds[0].shape[0])

    aggregate = list()
    for batch in iterate_in_chunks(pclouds, batch_size):
        if len(batch) == batch_size:
            probs = inference(sess, ops, batch, batch_size=batch_size)
            print np.argmax(probs, axis=1)
            aggregate.append(probs[:, class_index])

    print np.mean(aggregate)
