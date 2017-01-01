import tensorflow as tf
import numpy as np

batch_size = 24
Npoint = 2700
learning_rate = 0.01
training_epochs = 1000
trX = []
trY = []

def readPointFile(filename):
    pointlist = np.zeros([Npoint,3])
    numpoint = 0
    for l in open(filename):
        if not l.startswith('#'):
            x,y,z,leaveid = map(float,l.strip().split())
            pointlist[numpoint][0] = x
            pointlist[numpoint][1] = y
            pointlist[numpoint][2] = z
            numpoint = numpoint + 1
    return pointlist

def load_data():
    filelist = '/orions4-zfs/projects/lins2/Panos_Space/DATA/SN_point_clouds_Eric_annotated/gold_1K_with_all_parts.txt'
    for line in open(filelist):
        filename,Part = line.strip().split()
        filename = '/orions4-zfs/projects/lins2/Panos_Space/DATA/SN_point_clouds_Eric_annotated/03001627/bootstrapped_2p7K/' + filename + '_segs.txt'
        pointcloud = readPointFile(filename)
        trX.append(pointcloud)
        trY.append(pointcloud)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def model(X,w_1,w_2,w_3,w_4,w_5):
    X_ = tf.reshape(X,[-1,Npoint*3])
    h1 = tf.nn.relu(tf.matmul(X_,w_1))
    h2 = tf.nn.relu(tf.matmul(h1,w_2))
    h3 = tf.nn.relu(tf.matmul(h2,w_3))
    h4 = tf.nn.relu(tf.matmul(h3,w_4))
    h5 = tf.nn.sigmoid(tf.matmul(h4,w_5))
    out = tf.reshape(h5,[-1,Npoint,3])
    return out

X = tf.placeholder("float",[None, Npoint, 3])
Y = tf.placeholder("float",[None, Npoint, 3])

w_1 = init_weights([Npoint*3,Npoint*4])
w_2 = init_weights([Npoint*4,Npoint])
w_3 = init_weights([Npoint,int(Npoint*0.2)])
w_4 = init_weights([int(Npoint*0.2),Npoint])
w_5 = init_weights([Npoint,Npoint*3])

# Construct a linear model
pred = model(X,w_1,w_2,w_3,w_4,w_5)

# Mean Sequared Error
cost = tf.reduce_mean(tf.pow(pred-Y,2)/batch_size)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    # you  need to initalize all variables
    tf.initialize_all_variables().run()

    load_data()
    for i in range(1000):
        training_batch = zip(range(0,len(trX),batch_size),range(batch_size,len(trX)+1,batch_size))
        for start,end in training_batch:
            sess.run(optimizer,feed_dict={X:trX[start:end],Y:trY[start:end]})
            print(sess.run(pred,feed_dict={X:trX[start:end]}))
