'''
Created on Mar 4, 2017

@author: optas
'''
import tensorflow as tf
import time

from . point_net_ae import PointNetAutoEncoder
from . in_out import apply_augmentations
from .. fundamentals.loss import Loss

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
                self.out = tf.reshape(layer, [-1, c.n_output[0], c.n_output[1] + c.n_extra_pred[1]])
                out = self.out
                self.x_reconstr = out[:, :, :c.n_output[1]]
                self.extra_preds = out[:, :, c.n_output[1]:]

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

            print self.gt
            print self.x_reconstr

            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        if c.n_extra_pred is not None:
            extra_loss = tf.reduce_mean(Loss.cosine_distance_loss(self.extra_preds, self.extra_preds_gt))
            self.loss = self.loss + c.relative_loss_weight * extra_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=c.learning_rate).minimize(self.loss)

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
                loss, _ = self.partial_fit(batch_i, gt_data, extra_pred)
            else:
                loss, _ = self.partial_fit(batch_i, gt_data)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    def partial_fit(self, X, GT, extra_pred=None):
        print type(extra_pred)
        print type(X)
        print type(GT)
        if extra_pred is not None:
            print self.sess.run([tf.shape(self.gt), tf.shape(self.x_reconstr), tf.shape(self.out)], feed_dict={self.x: X, self.gt: GT, self.extra_preds_gt: extra_pred})

            _, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X, self.gt: GT, self.extra_preds_gt: extra_pred})
        else:
            _, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: X, self.gt: GT})
        return loss
