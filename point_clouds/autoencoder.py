'''
Created on February 2, 2017

@author: optas
'''

import os.path as osp
import tensorflow as tf
from general_tools.in_out.basics import create_dir

model_saver_id = 'models.ckpt'


class AutoEncoder(object):
    '''
    '''
    def __init__(self, name):
        self.nane = name
        with tf.device('/cpu:0'), tf.name_scope(name):
            self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)

    def restore_model(self, model_path, epoch):
        '''Restore all the variables of the auto-encoder.
        '''
        self.saver.restore(self.sess, osp.join(model_path, model_saver_id + '-' + str(epoch)))
#         print self.global_step

    def partial_fit(self, X, GT=None):
        '''Train models based on mini-batch of input data.
        Returns cost of mini-batch.
        If the AE is de-noising the GT needs to be provided.
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
        # Training cycle
        c = configuration
        if c.saver_step is not None:
            create_dir(c.train_dir)
        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)
            self.epoch = tf.add(self.epoch, 1)
            epoch = self.epoch.eval()
            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
            # Save the models checkpoint periodically.
            if c.saver_step is not None and epoch % c.saver_step == 0:
                checkpoint_path = osp.join(c.train_dir, model_saver_id)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)
