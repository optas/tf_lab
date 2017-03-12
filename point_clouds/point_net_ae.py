'''
Created on January 26, 2017

@author: optas
'''

import time
import tensorflow as tf
import socket

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from general_tools.in_out.basics import create_dir

from . autoencoder import AutoEncoder
from . in_out import apply_augmentations
from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.loss import Loss

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
            layer = c.decoder(self.z, **c.decoder_args)

#                 n_output = c.n_input[0]
#                 mask = fully_connected(tf.reshape(self.x_reconstr, [-1, 1, np.prod(c.n_input)]), n_output, 'softmax', 
#                 self.consistent = tf.transpose(tf.multiply(mask, tf.transpose(self.x_reconstr, perm=[0,2,1])), perm=[0,2,1])

            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)
            self._create_loss_optimizer()

            # GPU configuration

            if hasattr(c, 'allow_gpu_growth'):  # TODO - mitigate hasaatr
                growth = c.allow_gpu_growth
            else:
                growth = False

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)
#         print 'Trainable Parameters = %d' % (.count_trainable_parameters(self.graph), )

    def _create_loss_optimizer(self):
        c = self.configuration
        if c.loss == 'l2':
            self.loss = Loss.l2_loss(self.x_reconstr, self.gt)
        elif c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        if hasattr(c, 'consistent_io') and c.consistent_io:  # TODO - mitigate hasaatr
            self.output_mask = fully_connected(self.x_reconstr, self.n_output[0], activation='softmax', weights_init='xavier', name='consistent-softmax')
            _, indices = tf.nn.top_k(self.output_mask, self.n_input[0], sorted=False)

            self.output_cons_subset = tf.gather_nd(self.x_reconstr, indices)
            print self.output_cons_subset
#             print selector
#             self.sess.run(tf.shape(selector))
# 
#             self.output_cons_subset = self.x_reconstr[:, selector, :]
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.output_cons_subset, self.x)
            self.cons_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
            self.loss += self.cons_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)   # rename to train_step

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
