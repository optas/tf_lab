'''
Created on Feb 12, 2018

@author: optas
'''

import time
import numpy as np
import tensorflow as tf
from tflearn import is_training

from general_tools.simpletons import iterate_in_chunks
from .. neural_net import Neural_Net


class General_CLF(Neural_Net):

    def __init__(self, name, in_signal, in_labels, logits, learning_rate, graph=None):

        Neural_Net.__init__(self, name, None)

        self.x = in_signal
        self.gt = in_labels
        self.init_learning_rate = learning_rate
        self.logits = logits
        self.prediction = tf.argmax(self.logits, axis=1)

        self.correct_pred = tf.equal(self.prediction, tf.cast(self.gt, tf.int64))
        self.avg_accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self._create_loss_optimizer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Initializing the tensor flow variables
        self.init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def _create_loss_optimizer(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.gt)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.init_learning_rate).minimize(self.loss)

    def partial_fit(self, X, GT):
        '''Trains the model with mini-batches of input data.'''
        try:
            is_training(True, session=self.sess)
            _, loss, logits = self.sess.run((self.optimizer, self.loss, self.logits), feed_dict={self.x: X, self.gt: GT})
            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return logits, loss

    def predict(self, X, gt_labels=None):
        feed_dict = {self.x: X}
        if gt_labels is None:
            avg_acc = tf.no_op()
        else:
            avg_acc = self.avg_accuracy
            feed_dict[self.gt] = gt_labels
        return self.sess.run((self.prediction, avg_acc), feed_dict)

    def _single_epoch_train(self, train_data, batch_size):
        n_examples = train_data.n_examples
        epoch_loss = 0.
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        # Loop over all batches
        for _ in xrange(n_batches):
            batch_i, labels = train_data.next_batch(batch_size)
            _, loss = self.partial_fit(batch_i, labels)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    @staticmethod
    def classify_feed(clf, feed, batch_size, labels):
        predictions = []
        avg_acc = 0.
        last_examples_acc = 0.     # keep track of accuracy on last batch which potentially is smaller than batch_size
        n_last = 0.

        n_pclouds = len(feed)
        if labels is not None:
            if len(labels) != n_pclouds:
                raise ValueError()

        n_batches = 0.0
        idx = np.arange(n_pclouds)

        for b in iterate_in_chunks(idx, batch_size):
            feed_in = feed[b]
            feed_labels = labels[b]
            batch_predictions, batch_acc = clf.predict(feed_in, gt_labels=feed_labels)
            predictions.append(batch_predictions)
            if len(b) == batch_size:
                avg_acc += batch_acc
            else:  # last index was smaller than batch_size
                last_examples_acc = batch_acc
                n_last = len(b)
            n_batches += 1

        if n_last == 0:
            avg_acc /= n_batches
        else:
            avg_acc = (avg_acc * batch_size) + (last_examples_acc * n_last)
            avg_acc /= ((n_batches - 1) * batch_size + n_last)

        return predictions, avg_acc
