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
            self.graph = tf.get_default_graph()     # TODO change to make a new graph.

        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, configuration)

        with tf.variable_scope(name):
            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            layer = c.decoder(self.z, **c.decoder_args)
            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):  # TODO - mitigate hasaatr
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def trainable_parameters(self):
        # TODO: what happens if more nets in single graph?
        return count_trainable_parameters(self.graph)

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

        if hasattr(c, 'consistent_io') and c.consistent_io is not None:  # TODO - mitigate hasaatr
            self.cons_loss = PointNetAutoEncoder._consistency_loss(self)
            self.loss += self.cons_loss

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, global_step=self.epoch, decay_steps=10, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.summaries.append(tf.scalar_summary('learning_rate', self.lr))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
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
                loss, _ = self.partial_fit(batch_i, original_data)
            else:
                loss, _ = self.partial_fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    def gradient_wrt_input(self, in_points, gt_points):
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})

    @staticmethod
    def _consistency_loss(self):   # TODO Make instance method.
        c = self.configuration
        batch_indicator = np.arange(c.batch_size, dtype=np.int32)   # needed to match mask with output.
        batch_indicator = batch_indicator.repeat(self.n_input[0])
        batch_indicator = tf.constant(batch_indicator, dtype=tf.int32)
        batch_indicator = tf.expand_dims(batch_indicator, 1)

        output_mask = fully_connected(self.x_reconstr, self.n_output[0], activation='softmax', weights_init='xavier', name='consistent-softmax')
        _, indices = tf.nn.top_k(output_mask, self.n_input[0], sorted=False)

        indices = tf.reshape(indices, [-1])
        indices = tf.expand_dims(indices, 1)
        indices = tf.concat(1, [batch_indicator, indices])

        self.output_cons_subset = tf.gather_nd(self.x_reconstr, indices)
        self.output_cons_subset = tf.reshape(self.output_cons_subset, [c.batch_size, -1, self.n_output[1]])

        if c.consistent_io.lower() == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.output_cons_subset, self.x)
            return tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.consistent_io.lower() == 'emd':
            match = approx_match(self.output_cons_subset, self.x)
            return tf.reduce_mean(match_cost(self.output_cons_subset, self.x, match))
        else:
            assert(False)
