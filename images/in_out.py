'''
Created on Feb 13, 2018

@author: optas
'''

import numpy as np
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
