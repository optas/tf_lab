'''
Created on Feb 13, 2018

@author: optas
'''

import numpy as np
import tensorflow as tf
from PIL import Image

# To visualize a numpy image-array.
# Image.fromarray(a.astype('uint8'))


def load_png_and_focus_on_center(image_file, target_w, target_h):
    img = Image.open(image_file)
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    half_target_w = target_w / 2
    half_target_h = target_h / 2

    img = img.crop((half_the_width - half_target_w, half_the_height - half_target_h,
                    half_the_width + half_target_w, half_the_height + half_target_h))

    return np.array(img)


def resize_image_keep_aspect(image, fixed_dim, force_min=True):
    """ 
    The height or the width of the resulting image will  be `fixed_dim`.
    If force_min, then the side that is smaller will become `fixed_dim`.  
    """
    
    # Take width/height
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    # Take the greater value, and use it for the ratio
    if force_min:    
        min_ = tf.minimum(initial_width, initial_height)
        ratio = tf.to_float(min_) / tf.constant(fixed_dim, dtype=tf.float32)
    else:
        max_ = tf.maximum(initial_width, initial_height)
        ratio = tf.to_float(max_) / tf.constant(fixed_dim, dtype=tf.float32)

    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)

    return tf.image.resize_images(image, [new_width, new_height])