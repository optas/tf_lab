import tensorflow as tf

from trainer import Trainer

class MultiGPUTrainer(Trainer) :
    def __init__(self, graph, inputs, labels, create_inference, optimizer, loss, gpus=[0], name="trainer") :
        with tf.device('/cpu:0'):
            Trainer.__init__(self, graph, name, create_inference, optimizer, loss )

            gpu_count = len(gpus)

                            # Split the incoming and outgoing tensor streams into seperate streams, one for each GPU.   They each
                            # have to be the same size!    This will throw an exception if it cannot be evenly divided.
            input_tensors = inputs.split(gpu_count)
            label_tensors = labels.split(gpu_count)

                            # Track the gradient from each tower (corresponding to the network instantiated to each GPU).
            tower_grads = []
            total_losses = []

                            # Iterate over each GPU and perform a single feed-forward batch and compute loss and gradients to apply.   Each GPU
                            # computation will occur in parallel with no inter-dependencies.
            for i in range(len(gpus)):
                with tf.device('/gpu:%d' % gpus[i]):
                    with tf.name_scope('Tower_%d' % (i)) as scope:
                                    # Feed-forward and compute the loss over the entire mini-batch.
                        x = input_tensors[i]
                        y = label_tensors[i].unwrap()

                        inf = self.create_inference(x).unwrap()
                        self.loss.compute_loss(inf, y)
                        tower_loss = self.loss.get_total_loss(scope)
                        total_losses.append(tf.expand_dims(tower_loss, 0))

                                    # We must tell TF that all the variables instantiated above in the inference must be re-used on successive
                                    # GPU Towers so that we are training and affecting the same variables across GPUs.
                        tf.get_variable_scope().reuse_variables()

                                    # Record the summaries from the last GPU Tower
                        self.summaries.extend( tf.get_collection(tf.GraphKeys.SUMMARIES, scope) )

                                    # Compute the gradients using the optimizer and group together into an array.
                        grads = optimizer.operator.compute_gradients(tower_loss)

                        grads = [(g, v) for g, v in grads if g is not None]
                        tower_grads.append(grads)

                            # Average the gradients across the towers.
            grads = self._average_gradients(tower_grads)

                            # Apply the gradients to the trainable variables (backpropogate)
            self.training_operator = optimizer.operator.apply_gradients(grads, global_step=self.global_step)
            self.gradients_operator = grads

            self.batch_size = inputs.batch_size


                            # Average losses across towers.
                            # Add summaries of the loss.

            # summarize total loss and set loss operator
            total_loss = tf.reduce_mean(tf.concat(0, total_losses))
            self._summarize_all_losses(total_loss)


    def _average_gradients(self, tower_grads):
      """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
      return average_grads
