'''
Created on February 2, 2017

@author: optas
'''

import os.path as osp
import tensorflow as tf

from general_tools.in_out.basics import create_dir

model_saver_id = 'models.ckpt'


class AutoEncoder(object):
    '''Basis class for a Neural Network that implements an Auto-Encoder in TensorFlow.
    '''

    def __init__(self, name):
        self.name = name
        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

    def restore_model(self, model_path, epoch):
        '''Restore all the variables of a saved auto-encoder model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, model_saver_id + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            raise IOError('Loading model failed.')
        else:
            print 'Model restored in epoch %d.' % (epoch,)

    def partial_fit(self, X, GT=None):
        '''Trains the model with mini-batches of input data.
        If the AE is de-noising the GT needs to be provided.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        if GT is not None:
            _, loss, recon = self.sess.run((self.optimizer, self.loss, self.x_reconstr), feed_dict={self.x: X, self.gt: GT})
        else:
            _, loss, recon = self.sess.run((self.optimizer, self.loss, self.x_reconstr), feed_dict={self.x: X})
        return loss, recon

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def reconstruct(self, X):
        '''Use AE to reconstruct given data.'''
        return self.sess.run((self.x_reconstr, self.loss), feed_dict={self.x: X, self.gt: X})

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

    def evaluate(self, in_data, configuration, get_original=True):
        n_examples = in_data.num_examples
        data_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        reconstructions = []
        original = []
        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, _, _ = in_data.next_batch(batch_size)
            batch_i = batch_i.reshape([batch_size] + configuration.n_input)
            rec_i, loss = self.reconstruct(batch_i)
            reconstructions.append(rec_i)
            if get_original:
                original.append(batch_i)
            # Compute average loss
            data_loss += loss
        data_loss /= n_batches
        return reconstructions, data_loss, original
