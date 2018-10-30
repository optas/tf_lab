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

    def __init__(self, name, in_signal, in_labels, logits, 
                 learning_rate=0.0, 
                 dataset_pl_names=['in_signal', 'in_labels'],
                 graph=None):

        Neural_Net.__init__(self, name, graph)
        
        self.dataset_map = dict()
        self.x = in_signal
        self.dataset_map[self.x] = dataset_pl_names[0]
        self.gt = in_labels        
        self.dataset_map[self.gt] = dataset_pl_names[1]
        
        self.logits = logits
        self.prediction = tf.argmax(self.logits, axis=1)
        self.correct_pred = tf.equal(self.prediction, tf.cast(self.gt, tf.int64))
        self.avg_accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.create_loss_optimizer(learning_rate)
        
        # Initializing the tensor flow variables
        self.init = tf.global_variables_initializer()

        # Launch a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def create_loss_optimizer(self, learning_rate):
        self.lr = tf.get_variable('learning_rate', trainable=False, initializer=learning_rate)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.gt)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def partial_fit(self, in_signal, in_labels):
        '''Trains the model with mini-batches of input data.'''
        _, loss = self.sess.run([self.optimizer, self.loss], 
                                feed_dict={self.x: in_signal, self.gt: in_labels})
        return loss

    def _single_epoch_train(self, train_data, batch_size):
        n_examples = train_data.n_examples
        epoch_loss = 0.
        n_batches = int(n_examples / batch_size)
        start_time = time.time()
        try:
            is_training(True, session=self.sess)
            # Loop over all batches
            for _ in xrange(n_batches):
                batch_i, labels = train_data.next_batch(batch_size)
                loss = self.partial_fit(batch_i, labels)                
                epoch_loss += loss
            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    
    def feed_dataset(self, dataset, batch_size):
        feed_dict = {}
        for b in iterate_in_chunks(np.arange(dataset.n_examples), batch_size):
            for k, v in self.dataset_map.iteritems():
                feed_dict[k] = dataset[v][b]
            yield feed_dict, b
        
        
    def evaluate_on_data(net, eval_tensors, dataset, batch_size=None):
        if batch_size is None:
            batch_size = dataset.n_examples
        res = []
        idx = []
        for feed, k in feed_dataset(net, dataset, batch_size):
            r = net.sess.run(eval_tensors, feed)
            res.append(r)
            idx.append(k)
        return res, idx