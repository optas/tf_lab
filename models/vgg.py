'''
Wrapping VGG (Image-CNN).
VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf
'''

import tensorflow as tf
import numpy as np
from functools import partial

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg


from .. neural_net import Neural_Net
from .. data_sets.image_net import mean_rgb as img_net_rgb
 
 
# 1. What happens when you use the tf.variable_scope twice in __init__? 
#
#

class VGG_Dataset():
    def __init__(self, input_dims=[224, 224, 3], 
                 input_dtype=tf.float32, name='vgg_16', graph=None):
        
        self.img_pl = tf.placeholder(input_dtype, [None] + input_dims, 'img_pl')
        self.label_pl = tf.placeholder(tf.int32, [None], 'label_pl')
        
        
        
class VGG_Net(Neural_Net):    
    
    imagenet_rgb = img_net_rgb # Tensorflow-official model was trained with image_net.
        
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
    
    def add_standard_clf_ops(self):
        self.prediction = tf.to_int32(tf.argmax(self.logits, 1))
        self.correct_prediction = tf.equal(self.prediction, self.label_pl)
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
        
    def restore_pre_trained_variables(self, checkpoint, include=None, exclude=None):
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=include, exclude=exclude)
        # Calling function `self.init_pretrain(sess)` will load all the pre-trained weights.
        self.init_pretrain = tf.contrib.framework.assign_from_checkpoint_fn(checkpoint, variables_to_restore)    
        
    def evaluate_tensor(self, in_img, in_label, tensor):
        ''' in_img, in_label = some_iterator.next()
        '''
        result = []
        while True:
            try:
                feed_dict = {self.in_img: in_img, self.in_labels: in_label}
                res = self.sess.run(tensor, feed_dict)
                result.append(res)
            except tf.errors.OutOfRangeError:
                break
        result = np.vstack(result)
        return result

    def extract_features(self, in_img, in_label, end_point='vgg_16/fc7'):
        feat_layer = self.end_points[end_point]
        result = self.evaluate_tensor(self, in_img, in_label, feat_layer)
        result = np.squeeze(result).astype(np.float32)
        return result
    
    
def make_dataset_from_filenames(img_files, img_labels, batch_size, image_loader, 
                                preprocess_func=None,
                                shuffle=False, img_channels=3, shuffle_buffer=10000):
    
    filenames = tf.constant(img_files)
    labels = tf.constant(img_labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    parse_func = image_loader
    dataset = dataset.map(parse_func)
    
    if preprocess_func is not None:
        dataset = dataset.map(preprocess_func)
        
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        
    dataset = dataset.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    return dataset, iterator
    

def select_vgg_type(vgg_type):
    if vgg_type == 16:
        net = vgg.vgg_16
    elif vgg_type == 19:
        net = vgg.vgg_19
    else:
        raise ValueError('vgg type is neither 16 or 19.')
    return net



def traditional_vgg_image_scale(image, smallest_side=256.0):
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


def load_image_from_drive(image_file, label, channels=3, traditional_vgg_scaling=False):
    '''
    Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py        
        # Preprocessing (for both training and validation):
        # (1) Decode the image from PNG format
        # (2) Resize the image so its smaller side is 256 pixels long
        
    We apply (2) if `traditional_vgg_scaling` is True.
    '''
    
    image_string = tf.read_file(image_file)   
    
    # If needed, the PNG-encoded image is transformed to match the requested number 
    # of color channels.
    image_decoded = tf.image.decode_png(image_string, channels=channels)
    
    image = tf.cast(image_decoded, tf.float32)
    if channels == 1: # input image is gray-scale, turn to pseudo-RGB        
        image = tf.tile(image, [1, 1, 3])
        
    if traditional_vgg_scaling:
        image = traditional_vgg_image_scale(image)
        
    return image, label



def preprocess_image(image, label, subtract_mean=True, crop=None, 
                     horizontal_flip=False,
                     random_crop=False):
    
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
 


def traditional_vgg_preprocess_func(training=True):
    
    ''' If training:
        (1) Take a random 224x224 crop to the scaled image
        (2) Horizontally flip the image with probability 1/2
        (3) Subtract the per color mean `img_net` mean        
    '''
    if training:
        return partial(preprocess_image, subtract_mean=True, crop=[224, 224], horizontal_flip=True, random_crop=True)    
    else:
        return partial(preprocess_image, subtract_mean=True, crop=[224, 224], horizontal_flip=False, random_crop=False)
        
    
    
    
    
     
