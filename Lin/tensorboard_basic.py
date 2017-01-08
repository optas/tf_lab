from __future__ import print_function
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./tmp/logs',one_hot=True)

#Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = './tmp/logs'

#mnist data image of shape 28*28 =  784
x = tf.placeholder(tf.float32,[None,784],name='InputData')
#0-9 digits recognition ==> 10 classes
y = tf.placeholder(tf.float32,[None,10],name='LabelData')

# set model weights
W = tf.Variable(tf.zeros([784,10]),name='Weights')
b = tf.Variable(tf.zeros([10]),name='Bias')

#construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    #Model
    pred = tf.nn.softmax(tf.matmul(x,W)+b) # Softmax

with tf.name_scope('Loss'):
    #Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

with tf.name_scope('SGD'):
    # Grdient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))

# Initlializing the variables
init = tf.initialize_all_variables()

# Create a summary to monitor cost tensor
tf.scalar_summary("loss",cost)
# Create a summary to monitor accuarcy tensor
tf.scalar_summary("accuarcy",acc)
# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()

saver = tf.train.Saver()

#Lauch the graph
with tf.Session() as sess:
    sess.run(init)

    #op to wirte logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(logs_path,graph=tf.get_default_graph())

    # Saving the variables to disk
    save_path = saver.save(sess,"./tmp/model.ckpt")
    print("Model saved in files:%s"  % save_path)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op
            _,c,summary = sess.run([optimizer,cost,merged_summary_op],feed_dict={x:batch_xs,y:batch_ys})

            summary_writer.add_summary(summary,epoch*total_batch+i)
            avg_cost += c / total_batch

        if (epoch+1) % display_step == 0 :
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))

        print("optimization finished!")
