'''
Created on January 26, 2017

@author: optas
'''

import time
import numpy as np
import tensorflow as tf
import socket
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from general_tools.in_out.basics import create_dir

from . autoencoder import AutoEncoder
from . in_out import apply_augmentations
from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.loss import Loss
from .. fundamentals.inspect import count_trainable_parameters

try:
    if socket.gethostname() == socket.gethostname() == 'oriong2.stanford.edu':
        from .. external.oriong2.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.oriong2.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
    else:
        from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded.')


class PointNetAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder replicating the architecture of Charles and Hao paper.
    '''

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            # g = tf.Graph()
            # with g.as_default():
            self.graph = tf.get_default_graph()

        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, configuration)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)

            if c.exists_and_is_not_none('do_completion'):                   # TODO Re-factor for AP
                self.completion = tf.reshape(layer, [-1, c.n_completion[0], c.n_completion[1]])
                self.x_reconstr = tf.concat(1, [self.x, self.completion])   # output is input + `completion`
            else:
                self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def trainable_parameters(self):
        return count_trainable_parameters(self.graph, name_space=self.name)

    def _create_loss(self):
        c = self.configuration

        if c.loss == 'l2':
            self.loss = Loss.l2_loss(self.x_reconstr, self.gt)
        elif c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration, only_fw=False):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if self.is_denoising:
                _, loss = fit(batch_i, original_data)
            else:
                _, loss = fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    def gradient_wrt_input(self, in_points, gt_points):
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})