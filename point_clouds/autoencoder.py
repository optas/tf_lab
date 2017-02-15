'''
Created on February 2, 2017

@author: optas
'''

import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from general_tools.in_out.basics import create_dir

model_saver_id = 'models.ckpt'


class Configuration():
    def __init__(self, n_input, training_epochs=200, batch_size=10, learning_rate=0.001, denoising=False, non_linearity=tf.nn.relu,
                 saver_step=None, train_dir=None, z_rotate=False, loss='l2', gauss_augment=None, decoder=None, encoder=None, saver_max_to_keep=None, loss_display_step=1,
                 spatial_trans=False, debug=False, n_z=None, latent_vs_recon=1.0):
        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        # Parameters for any AE
        self.n_input = n_input
        self.encoder_sizes = [64, 128, 1024]
        self.non_linearity = non_linearity
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.spatial_trans = spatial_trans
        self.debug = debug
        # Used in VAE
        self.latent_vs_recon = latent_vs_recon
        self.n_z = n_z

        def __str__(self):
            keys = self.__dict__.keys()
            vals = self.__dict__.values()
            index = np.argsort(keys)
            res = ''
            for i in index:
                res += '%30s: %s\n' % (str(keys[i]), str(vals[i]))
            return res


class AutoEncoder(object):
    '''Basis class for a Neural Network that implements an Auto-Encoder in TensorFlow.
    '''

    def __init__(self, name, n_input, is_denoising):
        self.name = name
        self.is_denoising = is_denoising

        in_shape = [None, n_input[0], n_input[1]]

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            if self.is_denoising:
                self.gt = tf.placeholder(tf.float32, in_shape)
            else:
                self.gt = self.x

            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

    def restore_model(self, model_path, epoch, verbose=False):
        '''Restore all the variables of a saved auto-encoder model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, model_saver_id + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print 'Model restored in epoch %d.' % (epoch,)

    def partial_fit(self, X, GT=None):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a de-noising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        if GT is not None:
            _, loss, recon = self.sess.run((self.optimizer, self.loss, self.x_reconstr), feed_dict={self.x: X, self.gt: GT})
        else:
            _, loss, recon = self.sess.run((self.optimizer, self.loss, self.x_reconstr), feed_dict={self.x: X})
        return loss, recon

    def reconstruct(self, X, GT=None):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        if GT is None:
            GT = X
        return self.sess.run((self.x_reconstr, self.loss), feed_dict={self.x: X, self.gt: GT})

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def train(self, train_data, configuration):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.epoch.assign_add(tf.constant(1.0))))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and epoch % c.saver_step == 0:
                checkpoint_path = osp.join(c.train_dir, model_saver_id)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)
        return stats

    def evaluate(self, in_data, configuration, return_feed=True):
        '''
        return_feed: if True, return also input batch data.
        '''
        n_examples = in_data.num_examples
        data_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        reconstructions = []
        feed = []
        gt_feed = []
        # Loop over all batches
        for _ in xrange(n_batches):
            gt_data, labels, noisy_data = in_data.next_batch(batch_size)
            gt_data = gt_data.reshape([batch_size] + configuration.n_input)

            if self.is_denoising:
                noisy_data = noisy_data.reshape([batch_size] + configuration.n_input)
                batch_i = noisy_data    # Feed the noisy-version of the gt_data.
                rec_i, loss = self.reconstruct(batch_i, gt_data)
            else:
                batch_i = gt_data
                rec_i, loss = self.reconstruct(batch_i)

            reconstructions.append([rec_i, labels])

            if return_feed:
                feed.append((batch_i, labels))
                if self.is_denoising:
                    gt_feed.append([gt_data, labels])

            # Compute average loss
            data_loss += loss
        data_loss /= n_batches
        return reconstructions, data_loss, feed, gt_feed
