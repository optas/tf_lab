'''
Wrapping VGG (Image-CNN).
VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf
Pre-processing:
https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
'''

import tensorflow as tf
import numpy as np
from functools import partial

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg

from general_tools.simpletons import iterate_in_chunks
        
from .. neural_net import Neural_Net
from .. data_sets.image_net import mean_rgb as img_net_rgb
 
 
# 1. What happens when you use the tf.variable_scope twice in __init__? 
# 2. Model-variables.
#

class VGG_Net(Neural_Net):    
    
    imagenet_rgb = img_net_rgb # Tensorflow's-official model was trained with image_net.
        
    def __init__(self, in_img, in_label, vgg_type=16, n_classes=1000, name='vgg_16', graph=None):
        
        Neural_Net.__init__(self, name, graph)
        
        with self.graph.as_default():
            # Add the basic building blocks of  the network in the graph.
            with tf.variable_scope(name) as scope:
                net_builder = select_vgg_type(vgg_type)
                self.in_img = in_img
                self.in_label = in_label
                self.dropout_keep_prob = tf.get_variable('dropout_keep_prob', [], tf.float32, trainable=False)
                self.weight_decay = tf.get_variable('l2_weight_decay', [], tf.float32, trainable=False)
                self.is_training = tf.get_variable('is_training', [], tf.bool, trainable=False)

                
    
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=self.weight_decay)):            
                self.logits, self.end_points = net_builder(self.in_img, 
                                                           num_classes=n_classes,
                                                           is_training=self.is_training,
                                                           dropout_keep_prob=self.dropout_keep_prob,
                                                           scope=scope)
    
    
    def make_pre_trained_var_init(self, checkpoint, include=None, exclude=None):
        self.pre_trained_vars = tf.contrib.framework.get_variables_to_restore(include=include, exclude=exclude)
        # Calling function `self.init_pretrain(sess)` will load all the pre-trained weights.
        self.init_pretrain = tf.contrib.framework.assign_from_checkpoint_fn(checkpoint, self.pre_trained_vars)
        
    def make_fine_tune_var_init(self, var_names=[]):
        var_list = []
        for name in var_names:
            var_list.extend(tf.contrib.framework.get_variables(name))
        
        self.ft_vars = var_list
        self.init_finetune = tf.variables_initializer(self.ft_vars)
    
    def add_x_entropy_loss(self):
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection        
        tf.losses.sparse_softmax_cross_entropy(labels=self.in_label, logits=self.logits)
        # We can then call the total loss easily (it adds all tf.GraphKeys.LOSSES)
        self.loss = tf.losses.get_total_loss()
    
    def create_optimizer(self, full_lr, ft_lr=0):
        if len(self.ft_vars) > 0:
            self.ft_optimizer = tf.train.GradientDescentOptimizer(ft_lr)
            self.ft_train_step = self.ft_optimizer.minimize(self.loss, var_list=self.ft_vars)
                
        self.full_optimizer = tf.train.GradientDescentOptimizer(full_lr)
        self.full_train_step = self.full_optimizer.minimize(self.loss)
                        
    def add_standard_clf_ops(self):
        self.prediction = tf.to_int32(tf.argmax(self.logits, 1))
        self.correct_prediction = tf.equal(self.prediction, self.in_label)
        self.avg_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.probabilities = tf.nn.softmax(self.logits)
    
    def add_gradient_saliency(self):
        max_logit = tf.reduce_max(self.logits, axis=-1)
        saliency = tf.abs(tf.gradients(max_logit, self.in_img)[0])
        input_channels = self.in_img.shape.as_list()[-1]            
        if input_channels > 1:
            saliency = tf.reduce_sum(tf.abs(tf.gradients(max_logit, self.in_img)[0]), -1)
            norms = tf.reduce_sum(saliency, [1, 2], keep_dims=True) + 10e-12
            saliency /= norms
        self.saliency = saliency
            
    def evaluate_tensor_with_placeholders(self, tensor, in_img, in_label=None, batch_size=128):
        ''' TODO:make a tf.dataset from the placeholders'''
        result = []
        for b in iterate_in_chunks(np.arange(len(in_img)), batch_size):
            feed_dict = {self.in_img: in_img[b]} # Copying can be replaced with view. 
            if in_label is not None:
                feed_dict[self.in_label] = in_label[b]
            res = self.sess.run(tensor, feed_dict)
            result.append(res)
        try:
            result = np.vstack(result)
        except:
            result = np.hstack(result)
        return result
    
    def evaluate_tensor(self, tensor):
        ''' in_img, in_label = some_iterator.next()
        '''
        result = []       
        while True:
            try:
                res = self.sess.run(tensor)
                result.append(res)
            except tf.errors.OutOfRangeError:
                break
        try:
            result = np.vstack(result)
        except:
            result = np.hstack(result)
        
        return result

    def extract_features(self, in_img=None, in_label=None, end_point='vgg_16/fc7'):
        feat_layer = self.end_points[end_point]
        result = self.evaluate_tensor(feat_layer)
        result = np.squeeze(result).astype(np.float32)
        return result
    
    def clf_accuracy(self):
        num_correct, num_samples = 0, 0
        while True:
            try:
                correct_pred = self.sess.run(self.correct_prediction, {self.is_training: False})
                num_correct += correct_pred.sum()
                num_samples += correct_pred.shape[0]
            except tf.errors.OutOfRangeError:
                break
        # Return the fraction of datapoints that were correctly classified
        acc = float(num_correct) / num_samples
        return acc
        
    def train_ft(self, n_epochs, train_iter_init):
        for epoch in range(n_epochs):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, n_epochs))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.   
            self.sess.run(train_iter_init)
            while True:
                try:
                    self.sess.run(self.ft_train_step, {self.is_training: True})
                except tf.errors.OutOfRangeError:
                    break
            
            # Check accuracy on the train sets every epoch.
            self.sess.run(train_iter_init)
            train_acc = self.clf_accuracy()
            print('Train accuracy: %f' % train_acc)           
            
    def train_all(self, n_epochs, train_iter_init):
        for epoch in range(n_epochs):
            print('Starting epoch %d / %d' % (epoch + 1, n_epochs))
            self.sess.run(train_iter_init)
            while True:
                try:
                    self.sess.run(self.full_train_step, {self.is_training: True})
                except tf.errors.OutOfRangeError:
                    break
            
            # Check accuracy on the train sets every epoch.
            self.sess.run(train_iter_init)
            train_acc = self.clf_accuracy()
            print('Train accuracy: %f' % train_acc)
            
    
def make_dataset_from_filenames(img_files, img_labels, batch_size, image_loader, 
                                preprocess_func=None,
                                shuffle=False, img_channels=3, num_parallel_calls=6, shuffle_buffer=10000):
    '''
    TODO: read effect of shuffle_buffer size.
    '''
    filenames = tf.constant(img_files)
    labels = tf.constant(img_labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.map(image_loader, num_parallel_calls=num_parallel_calls)
    
    if preprocess_func is not None:
        dataset = dataset.map(preprocess_func, num_parallel_calls=num_parallel_calls)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset

    
def select_vgg_type(vgg_type):
    if vgg_type == 16:
        net = vgg.vgg_16
    elif vgg_type == 19:
        net = vgg.vgg_19
    else:
        raise ValueError('vgg type is neither 16 or 19.')
    return net


def load_image_from_drive(image_file, label, channels=3):
    '''Decode the image from PNG format.
    '''
    
    image_string = tf.read_file(image_file)   
    
    # If needed, the PNG-encoded image is transformed to match the requested number 
    # of color channels.
    image_decoded = tf.image.decode_png(image_string, channels=channels)
    
    image = tf.cast(image_decoded, tf.float32)
    if channels == 1: # input image is gray-scale, turn to pseudo-RGB        
        image = tf.tile(image, [1, 1, 3])
                
    return image, label


def preprocess_image(image, label, subtract_mean=True, crop=None, 
                     horizontal_flip=False,
                     random_crop=False, rescaler=None):
    
    if rescaler is not None:
        image = rescaler(image)
        
    if crop is not None:
        if random_crop:
            image = tf.random_crop(image, [crop[0], crop[1], 3])
        else:        
            image = tf.image.resize_image_with_crop_or_pad(image, crop[0], crop[1])
        
    if horizontal_flip:
        image = tf.image.random_flip_left_right(image)
    
    if subtract_mean:
        means = tf.reshape(tf.constant(img_net_rgb), [1, 1, 3])        
        image -= means
        
    return image, label

    
def standard_vgg_preprocess_func(training=True):
    ''' Standard preprocessing for VGG on ImageNet taken from here:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py  
    
    If training:
        (1) Resize the image so its smaller side is 256 pixels long
        (2) Take a random 224x224 crop of the scaled image
        (3) Horizontally flip the image with probability 1/2
        (4) Subtract the per color mean `img_net` mean
    '''
    rescaler = standard_vgg_image_scale
    if training:
        return partial(preprocess_image, subtract_mean=True, crop=[224, 224], horizontal_flip=True, random_crop=True, rescaler=rescaler)
    else:
        return partial(preprocess_image, subtract_mean=True, crop=[224, 224], horizontal_flip=False, random_crop=False)

    
def standard_vgg_image_scale(image, smallest_side=256.0):
    ''' Input transformation applied in all image-net images.
    '''
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])
    return resized_image