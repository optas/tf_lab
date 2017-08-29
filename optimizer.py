'''
Created on Aug 28, 2017

@author: optas
'''
import tensorflow as tf
from . fundamentals.inspect import add_gradients_summary


class Optimizer(object):

    def __init__(self, tf_optimizer, tf_optimizer_kwargs, loss, var_list, transform_grads=None):
        '''
        TODO. global_step
        '''
        self.optimizer = tf_optimizer(**tf_optimizer_kwargs)
        self.grads_and_vars = self.optimizer.compute_gradients(loss, var_list=var_list)

        if transform_grads is not None:
            self.grads_and_vars = transform_grads(self.grads_and_vars)

        self.opt_step = self.optimizer.apply_gradients(self.grads_and_vars)

        add_gradients_summary(self.grads_and_vars)

    @staticmethod
    def capp_grad_by_value(grad_and_var, lbound, r_bound):
        return [(tf.clip_by_value(grad, lbound, r_bound), var) for grad, var in grad_and_var]
