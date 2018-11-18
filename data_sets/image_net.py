'''
Created on Nov 17, 2018

@author: optas
'''

import numpy as np
import ast


mean_rgb = np.array([123.68, 116.78, 103.94], dtype=np.float32)


def int_to_class_name(file_with_dict): 
    with open(file_with_dict, 'r') as in_file:
        return ast.literal_eval(in_file.read())