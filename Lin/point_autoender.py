import tensorflow as tf
import numpy as np
import os
import time
from global_variables import *
import tensorflow.contrib.slim as slim
from load_data import readPointFile,load_data
from records_converter import read_and_decode
import fc_model
from visualizer import gen_plot

# Basic model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir','/orions4-zfs/projects/lins2/Lin_Space/DATA/Lin_Data/TFRecords','Directory with the training data.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')


Chair_FILE = 'chair.tfrecords'

def inputs(batch_size,num_epochs):
    """ Reads input data num_epochs times.

    Args:
        train: Selects between the training (True) and validation
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data or 0/None to train forever.

    Returns:
        A tuple
    """

    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.train_dir,Chair_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                [filename],num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename queue
        point_cloud = read_and_decode(filename_queue)
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        point_clouds = tf.train.shuffle_batch(
                [point_cloud], batch_size=batch_size,num_threads=4,
                capacity=1000 + 3 * batch_size,
                # Ensure a minimum amount of shuffling of examples
                min_after_dequeue=100)
        return point_clouds

# Add an op to initialize the variables.
def run_training():
    """Train for a number of steps."""

    # Tell Tensorflow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input point_cloud
        pcs = inputs(batch_size=batch_size,num_epochs=training_epochs)


        # Construct a linear model
        pred = fc_model.autoendoder(pcs)
        # Mean Sequared Error
        cost = fc_model.loss(pred,pcs)

        #
        gt = pcs
        # Create a variable to track the global step.
        global_step = tf.Variable(0,name='global_step',trainable=False)

        # Gradient descent
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

        #Add the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        #Create a saver for writing training checkpoints
        saver = tf.train.Saver()

        # The op for initializing the variables
        init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Create a session for running operations in the Graph.
        sess = tf.Session(config=config)

        # Initialize the variables (the trained variables and the epoch counter)
        sess.run(init_op)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,sess.graph)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _,loss_value,pred_,gt_ = sess.run([train_op,cost,pred,gt])
                duration = time.time() - start_time
                # Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    print('Step %d: loss = %.5f (%.3f sec)' % (step,loss_value,duration))

                    #Update the event  file.
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str,step)
                    step += 1
                    gen_plot(pred_[0],'test',step)
                    gen_plot(gt_[0],'gt',step)
                # Save a checkpoint periodically.
                if (step + 1) % 1000 == 0:
                    print('saving')
                    saver.save(sess,FLAGS.train_dir,global_step=step)

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,step))
            print('Saving')
            saver.save(sess,FLAGS.train_dir,global_step=step)
            print('Done training for %d epochs, %d step' % (FLAGS.num_epochs,step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish
        coord.join(threads)
        sess.close()

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
