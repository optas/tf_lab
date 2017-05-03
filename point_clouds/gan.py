'''
Created on May 3, 2017

@author: optas
'''

import os.path as osp
import warnings
import tensorflow as tf

from general_tools.in_out.basics import create_dir


model_saver_id = 'models.ckpt'


class GAN(object):
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                self.epoch = tf.get_variable('epoch', [], initializer=tf.constant_initializer(0), trainable=False)
        
        

    def restore_model(self, model_path, epoch, verbose=False):
        '''Restore all the variables of a saved model.
        '''
        self.saver.restore(self.sess, osp.join(model_path, model_saver_id + '-' + str(int(epoch))))

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn('Loaded model\'s epoch doesn\'t match the requested one.')
        else:
            if verbose:
                print('Model restored in epoch {0}.'.format(epoch))

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
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, model_saver_id)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

        return stats
