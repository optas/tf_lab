'''
Created on Mar 4, 2017

@author: optas
'''
import time
import tensorflow as tf
import numpy as np
import os.path as osp
model_saver_id = 'models.ckpt'

from general_tools.in_out.basics import create_dir


from . point_net_ae import PointNetAutoEncoder
from . in_out import apply_augmentations
from .. fundamentals.loss import Loss
from boto import config

try:
    from .. external.Chamfer_EMD_losses.tf_nndistance import nn_distance
    from .. external.Chamfer_EMD_losses.tf_approxmatch import approx_match, match_cost
except:
    print 'External Losses (Chamfer-EMD) cannot be loaded.'


class PointNetAbstractorPredictor(PointNetAutoEncoder):

    def __init__(self, name, configuration, graph=None):
        if graph is None:
            self.graph = tf.get_default_graph()

        c = configuration
        self.configuration = c
        self.name = name
        self.is_denoising = True
        self.n_input = c.n_input
        self.n_output = c.n_output

        in_shape = [None] + self.n_input
        out_shape = [None] + self.n_output

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            self.gt = tf.placeholder(tf.float32, out_shape)
            self.z = c.encoder(self.x, **c.encoder_args)
            layer = c.decoder(self.z, **c.decoder_args)

            if c.n_extra_pred is None:
                self.x_reconstr = tf.reshape(layer, [-1, c.n_output[0], c.n_output[1]])
            else:
                self.extra_preds_gt = tf.placeholder(tf.float32, [None] + c.n_extra_pred)
                out = tf.reshape(layer, [-1, c.n_output[0], c.n_output[1] + c.n_extra_pred[1]])
                self.x_reconstr = out[:, :, :c.n_output[1]]
                self.extra_preds = out[:, :, c.n_output[1]:]

#             if c.consistent_io:
#                 n_output = c.n_input[0]
#                 mask = fully_connected(tf.reshape(self.x_reconstr, [-1, 1, np.prod(c.n_input)]), n_output, 'softmax', weights_init='xavier', name='consistent')
#                 self.consistent = tf.transpose(tf.multiply(mask, tf.transpose(self.x_reconstr, perm=[0,2,1])), perm=[0,2,1])

        with tf.device('/cpu:0'):
            self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        self.init = tf.global_variables_initializer()

        # GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Launch the session
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def _create_loss_optimizer(self):
        c = self.configuration
        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        if c.n_extra_pred is not None:
            self.extra_pred_loss = c.relative_loss_weight * tf.reduce_mean(match_cost(self.extra_preds, self.extra_preds_gt, match))
#             self.extra_pred_loss = c.relative_loss_weight * tf.reduce_mean(Loss.cosine_distance_loss(self.extra_preds, self.extra_preds_gt))
            self.loss += self.extra_pred_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)

    def train(self, train_data, configuration):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)

            if len(loss) == 2:
                print ('Normal prediction loss = '  "{:.9f}".format(loss))
                loss = loss[0]

            epoch = int(self.sess.run(self.epoch.assign_add(tf.constant(1.0))))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, model_saver_id)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)
        return stats

    def _single_epoch_train(self, train_data, configuration):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        c = configuration
        
        for _ in xrange(n_batches):
            gt_data, _, batch_i = train_data.next_batch(batch_size)
            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if c.n_extra_pred is not None:
                extra_pred = gt_data[:, :, c.n_output[1]:]
                gt_data = gt_data[:, :, :c.n_output[1]]
                loss = self.partial_fit(batch_i, gt_data, extra_pred)
            else:
                loss = self.partial_fit(batch_i, gt_data)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    def partial_fit(self, X, GT, extra_pred=None):
        if extra_pred is not None:
            _, loss, pred_loss = self.sess.run((self.optimizer, self.loss, self.extra_pred_loss), feed_dict={self.x: X, self.gt: GT, self.extra_preds_gt: extra_pred})
            loss = (loss, pred_loss)
        else:
            _, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X, self.gt: GT})
        return loss

    def reconstruct(self, X, GT, extra_pred=None):
        feed_dict = {self.x: X, self.gt: GT, self.extra_preds_gt: extra_pred}
        if extra_pred is not None:
            return self.sess.run((self.x_reconstr, self.loss, self.extra_preds), feed_dict=feed_dict)
        else:
            return self.sess.run((self.x_reconstr, self.loss), feed_dict=feed_dict)

    def evaluate(self, in_data, configuration):
        n_examples = in_data.num_examples
        data_loss = 0.
        c = configuration
        original_data, ids, feed_data = in_data.full_epoch_data(shuffle=False)
        feed_data = apply_augmentations(feed_data, c)  # This is a new copy of the batch.
        b = c.batch_size
        reconstructions = np.zeros([n_examples] + c.n_output)

        if c.n_extra_pred is not None:
            gt_points = original_data[:, :, :c.n_output[1]]
            gt_extra_pred = original_data[:, :, c.n_output[1]:]
            extra_preds_recon = np.zeros([n_examples] + c.n_extra_pred)
            for i in xrange(0, n_examples, b):
                reconstructions[i:i + b], loss, extra_preds_recon[i:i + b] = self.reconstruct(feed_data[i:i + b], gt_points[i:i + b], gt_extra_pred[i:i + b])
                data_loss += (loss * len(reconstructions[i:i + b]))
        else:
            raise NotImplementedError()
            for i in xrange(0, n_examples, b):
                reconstructions[i:i + b], loss = self.reconstruct(feed_data[i:i + b], original_data[i:i + b])
                data_loss += (loss * len(reconstructions[i:i + b]))

        data_loss /= float(n_examples)
        return reconstructions, data_loss, np.squeeze(feed_data), ids, np.squeeze(original_data), np.squeeze(extra_preds_recon)
