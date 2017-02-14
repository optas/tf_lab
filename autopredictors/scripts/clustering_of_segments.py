'''
Created on December 22, 2016

@author: optas
'''

import numpy as np
import os
import glob
import os.path as osp
from collections import defaultdict
from sklearn.cluster import KMeans

from . helper import feat_ending

def load_AP_features(feature_dir):
    model_to_feat_id = defaultdict(dict)
    n_samples = 0
    feat_list = []
    model_list = []
    part_list = []
    counter = 0
    for file_name in glob.glob(osp.join(feature_dir, '*' + feat_ending)):        
        file_sig = osp.basename(file_name).rstrip()        
        file_sig = file_sig.split('_')
        model_name, part_id = file_sig[0], int(file_sig[1])    
        in_feat = np.loadtxt(file_name, skiprows=2)
        feat_list.append(in_feat)
        model_list.append(model_name)
        part_list.append(part_id)
        model_to_feat_id[model_name][part_id] = counter
        counter += 1
    if counter == 0:
        raise ValueError('No files loaded.')
    
    feat_array = np.array(feat_list)    
    part_array = np.array(part_list)
    model_array = np.array(model_list)
    return feat_array, part_array, model_array, model_to_feat_id  
            
        