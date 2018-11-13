'''
Created on Feb 12, 2018

@author: optas
'''

import tensorflow as tf
import numpy as np
import time
from tflearn import is_training
from general_tools.simpletons import iterate_in_chunks
from .. neural_net import Neural_Net


class Generic_CLF(Neural_Net):

    def __init__(self, name, in_signal, in_labels, logits,                 
                 dataset_pl_names=['in_signal', 'in_labels'],
                 graph=None):

        Neural_Net.__init__(self, name, graph)
        
        self.dataset_map = dict()
        self.x = in_signal
        self.dataset_map[self.x] = dataset_pl_names[0]
        self.gt = in_labels        
        self.dataset_map[self.gt] = dataset_pl_names[1]
        self.logits = logits
        
        with self.graph.as_default():
            with tf.variable_scope(name):
                self.prediction = tf.argmax(self.logits, axis=1)
                self.correct_pred = tf.equal(self.prediction, tf.cast(self.gt, tf.int64))
                self.avg_accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
                self.probabilities = tf.nn.softmax(self.logits)
                self._add_loss_and_optimizer()                
                self.model_vars = [g for g in tf.global_variables() if g.name.startswith(name)]
                self.init = tf.variables_initializer(self.model_vars)
                
                # Launch a session
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                self.sess.run(self.init)

    def _add_loss_and_optimizer(self):
        self.lr = tf.get_variable('learning_rate', trainable=False, shape=())
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
        
        with self.graph.as_default():            
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
        
        
    def evaluate_on_data(self, eval_tensors, dataset, batch_size=None):
        if batch_size is None:
            batch_size = dataset.n_examples
        res = []
        idx = []
        for feed, k in self.feed_dataset(dataset, batch_size):
            r = self.sess.run(eval_tensors, feed)
            res.append(r)
            idx.append(k)
        return res, idx
    
    def report_clf_accuracy(self, dataset, batch_size=None):        
        scores, b_sizes = self.evaluate_on_data(self.avg_accuracy, dataset, batch_size=batch_size)
        total_acc = 0.0
        for s, b in zip(scores, b_sizes):
            total_acc += (s * len(b))        
        return total_acc / dataset.n_examples