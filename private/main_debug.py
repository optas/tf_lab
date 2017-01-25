import sys
import numpy as np
import os.path as osp
import tensorflow as tf

git_path = '/Users/optas/Documents/Git_Repos/'
sys.path.insert(0, git_path)

from tf_lab.point_clouds import point_cloud_ae as pae
from tf_lab.point_clouds.point_cloud_ae import Configuration as ae_conf
import tf_lab.point_clouds.in_out as pio
from tf_lab.fundamentals.loss import Loss
from tf_lab.fundamentals.inspect import hist_summary_of_trainable, sparsity_summary_of_trainable

if __name__ == '__main__':
    config = ae_conf()

    in_signal, gt_signal = pio.in_out_placeholders(config)

    model = pae.autoencoder_with_fcs_only(in_signal, config)

    loss = Loss.l2_loss(model, gt_signal)

    optimizer = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(loss)
