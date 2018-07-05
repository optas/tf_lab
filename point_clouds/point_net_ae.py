'''
Created on January 26, 2017

@author: optas
'''

import time
import tensorflow as tf
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from general_tools.in_out.basics import create_dir

from . autoencoder import AutoEncoder
from . in_out import apply_augmentations
from . spatial_transformer import transformer as pcloud_spn
from .. fundamentals.loss import Loss
from .. fundamentals.inspect import count_trainable_parameters

from .. external.structural_pc_losses import losses
nn_distance, approx_match, match_cost = losses()

#import numpy as np
#from . utils import pdist
#from with_others.million_geometries.src.hacks import penalize_nn_of_two_embeddings

class PointNetAutoEncoder(AutoEncoder):
    '''
    '''
    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        AutoEncoder.__init__(self, name, graph, configuration)

        with tf.variable_scope(name) as scope:
                        
            #c.encoder_args['scope'] = scope
            #c.encoder_args['reuse'] = False
            #c.encoder_args['spn'] = True 
            #self.container = c.encoder_args['container']
            self.z = c.encoder(self.x, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
                        
            #self.x_r = self.container['signal_transformed']            
            #c.encoder_args['spn'] = False
            #c.encoder_args['reuse'] = True            
            #self.z_r = c.encoder(self.x_r, **c.encoder_args)
                        
            #c.decoder_args['scope'] = scope
            #c.decoder_args['reuse'] = False
            layer = c.decoder(self.z, **c.decoder_args)
            self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
            
            #c.decoder_args['reuse'] = True
            #layer = c.decoder(self.z_r, **c.decoder_args)
            #self.x_r_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])       
            #self.shrinkage_loss = 0.01 * penalize_nn_of_two_embeddings(self.z, self.z_r)

            if False:
                if c.exists_and_is_not_none('close_with_tanh'):
                    print('Closing decoder with tanh.')
                    layer = tf.nn.tanh(layer)

                if c.exists_and_is_not_none('do_completion'):                   # TODO Re-factor for AP
                    self.completion = tf.reshape(layer, [-1, c.n_completion[0], c.n_completion[1]])
                    self.x_reconstr = tf.concat([self.x, self.completion], axis=1)   # output is input + `completion`                
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
            #cost_p1_p2_, _, cost_p2_p1_, _ = nn_distance(self.x_r_reconstr, self.x_r)
            
            if c.exists_and_is_not_none('loss_reduction'):
                if c.loss_reduction == 'log_sum_exp':
                    self.loss = tf.reduce_logsumexp(cost_p1_p2, 1) + tf.reduce_logsumexp(cost_p2_p1, 1)
                elif c.loss_reduction == 'euclid_sqrt':
                    self.loss = tf.reduce_sum(tf.sqrt(cost_p1_p2) + tf.sqrt(cost_p2_p1), 1)
                else:
                    assert(False)
                self.loss = tf.reduce_mean(self.loss)
            else:
                self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
                #self.cost_p1_p2 = cost_p1_p2
                #self.loss_1 = tf.reduce_mean(cost_p1_p2, axis=1) + tf.reduce_mean(cost_p2_p1, axis=1)
                #self.loss_2 = tf.reduce_mean(cost_p1_p2_, axis=1) + tf.reduce_mean(cost_p2_p1_, axis=1)                                
                
                #rel = self.loss_2 - self.loss_1
                #rel = rel * tf.cast(rel > 0, tf.float32)
                #self.rel = rel
                #self.loss = self.loss_2 + rel
                #self.loss = tf.reduce_mean(self.loss_2 )
                
                
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
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)    
        
        #sensing_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/spn')
        #g_sr = tf.gradients(self.shrinkage_loss, sensing_vars)

        #for g, v in zip(g_sr, sensing_vars):
            #if g is not None:
                #self.grads_and_vars.append((g, v))
        
        #self.train_step = self.optimizer.apply_gradients(self.grads_and_vars)
        
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

            batch_i = apply_augmentations(batch_i, configuration) # This is a new copy of the batch.
            if self.is_denoising:
                _, loss = fit(batch_i, original_data)
            else:
                _, loss = fit(batch_i)
            
            # Compute average loss
            epoch_loss += loss
        epoch_loss /= float(n_batches)
        duration = time.time() - start_time        
        return epoch_loss, duration

    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})

    def gradient_of_input_wrt_latent_code(self, in_points, code_dims=None):
        ''' batching this is ok. but if you add a list of code_dims the problem is on the way the tf.gradient will
        gather the gradients from each dimension, i.e., by default it just adds them. This is problematic since for my
        research I would need at least the abs sum of them.
        '''
        b_size = len(in_points)
        n_dims = len(code_dims)

        row_idx = tf.range(b_size, dtype=tf.int32)
        row_idx = tf.reshape(tf.tile(row_idx, [n_dims]), [n_dims, -1])
        row_idx = tf.transpose(row_idx)
        col_idx = tf.constant(code_dims, dtype=tf.int32)
        col_idx = tf.reshape(tf.tile(col_idx, [b_size]), [b_size, -1])
        coords = tf.transpose(tf.pack([row_idx, col_idx]))

        if b_size == 1:
            coords = coords[0]
        ys = tf.gather_nd(self.z, coords)
        return self.sess.run(tf.gradients(ys, self.x), feed_dict={self.x: in_points})[0]
