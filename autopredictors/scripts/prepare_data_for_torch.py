'''Created on December 22, 2016

@author: optas
'''

import glob
import warnings
import numpy as np
import os.path as osp

from general_tools.arrays.is_true import is_contiguous
from . helper import segs_extension
from general_tools.in_out.basics import read_header_of_np_saved_txt
    

def read_pt_cloud_with_segs(in_file):
    in_data = np.loadtxt(in_file)
    pts = in_data[:,[0,1,2]]
    seg = in_data[:,3].astype(np.int32)
    return pts, seg
    
    
def read_model_id_to_max_segment(in_file):
    res = dict()
    with open(in_file) as f_in:
        for line in f_in:
            model_id, max_seg = line.rstrip().split(' ')
            res[model_id] = int(max_seg)
    return res


def write_model_id_to_max_segment(out_file, model_to_segs):
    with open(out_file, 'w') as f_out:
        for model_name, n_segs in model_to_segs.iteritems():
            f_out.write('%s %d\n' % (model_name, n_segs) )

    
def record_model_id_and_max_segments(segments_dir, out_file=None):
    model_to_segs = dict()
    for file_name in glob.glob(osp.join(segments_dir, '*_segs.txt')):
        model_name = osp.basename(file_name).split('_')[0]    
        _, seg = read_pt_cloud_with_segs(file_name)
        if not is_contiguous(seg):    
            warnings.warn('Loaded segmentation is not contiguous. Model = %s.' % (model_name, ) )
            
        else:
            seg_ids = np.unique(seg)
            if seg_ids[0] == 0:
                seg_ids = seg_ids[1:] # Zero is not a real segment.                      
            model_to_segs[model_name] = len(seg_ids)
    
    if out_file != None: # Write data to out_file.
        write_model_id_to_max_segment(out_file, model_to_segs)
                                              
    return model_to_segs
                    

def read_header_of_seg_file(file_name):
    header = read_header_of_np_saved_txt(file_name)
    feat_type, n_segs = header.splitlines()[:2]
    feat_type = feat_type.split('_')[0][2:]    
    n_segs = int(n_segs.split('=')[1])
    return feat_type, n_segs

                        
def generate_gold_standar(segs_dir, file_out, n_shapes='all', min_parts=None, max_parts=None, only_cc=True, only_hks=False, seed=0):
    valid_models = []    
    model_to_nsegs = dict()
    for file_name in glob.glob(osp.join(segs_dir, '*' + segs_extension)):
        model_name = osp.basename(file_name).split('_')[0]
        feat_type, n_segs = read_header_of_seg_file(file_name)
            
        if min_parts != None:
            if n_segs < min_parts:
                continue
        if max_parts != None:
            if n_segs > max_parts:
                continue
        if only_cc:        
            if feat_type != 'cc':
                continue 
        if only_hks:
            if feat_type != 'hks':
                continue             
                        
        valid_models.append(model_name)
        model_to_nsegs[model_name] = n_segs
        
    if n_shapes == 'all':
        picked_models = valid_models
        
    else:
        np.random.seed(seed)
        np.random.shuffle(valid_models)
        picked_models = valid_models[:n_shapes]    

    with open(file_out, 'w') as f_out:
        for m in picked_models:
            file_name = osp.join(segs_dir, m + segs_extension)
            n_segs = model_to_nsegs[m]    
            for s in range(1, n_segs + 1):
                f_out.write('%s %d\n' % (m, s))