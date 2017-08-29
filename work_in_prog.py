import tensorflow as tf
from tf_lab.fundamentals.inspect import summarize_gradients


class Optimizer(object):

    def __init__(self, tf_optimizer, tf_optimizer_kwargs, loss, var_list, summary_collection='tf_lab_summaries', transform_grads=None, name=''):
        self.optimizer = tf_optimizer(**tf_optimizer_kwargs)
        self.grads_and_vars = self.optimizer.compute_gradients(loss, var_list=var_list)
        
        if transform_grads is not None:
            self.grads_and_vars = transform_grads(self.grads_and_vars)

        self.opt_step = self.optimizer.apply_gradients(self.grads_and_vars)

        self.grad_sum = summarize_gradients(self.grads_and_vars, summary_collection)

    @staticmethod
    def capp_grad_by_value(grad_and_var, lbound, r_bound):
        return [(tf.clip_by_value(grad, lbound, r_bound), var) for grad, var in grad_and_var]
    
    
class Trainer(object):
    
    def __init__(self, nn_model, train_dir, graph=None):
        '''
        Constructor
        '''        
        if graph is None:
            self.graph = tf.get_default_graph()
        else:
            self.graph = graph
                
        self.train_dir = train_dir
        create_dir(train_dir)

        summary_collection = 'tf_lab_summaries'
        self.merged_summaries = tf.summary.merge(tf.get_collection(summary_collection))
        
        self.train_writer = tf.summary.FileWriter(osp.join(self.train_dir, 'summaries'), self.graph)
        
    def train(nn_model, sess, n_epochs, train_data):
        for epoch in xrange(n_epochs):
            loss, duration = nn_model.single_epoch_train(sess, train_data)
            summary = sess.run(nn_model.merged_summaries)
            self.train_writer.add_summary(summary, epoch)