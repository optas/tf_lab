from __future__ import print_function

import numpy as np
import sys
import os.path as osp

from tf_lab.nips.helper import center_pclouds_in_unit_sphere
from tf_lab.evaluate.generative_pc_nets import entropy_of_occupancy_grid, jensen_shannon_divergence, minimum_mathing_distance

def identity(x):
    return x

def sampling_mmd(sample_data, ref_data, n_samples=10, pop_size=100, emd=True, verbose=False):
    ids = np.arange(len(ref_data))
    scores = []
    for i in range(n_samples):
        if verbose:
            print(i)
        sb = np.random.choice(ids, pop_size, replace=False)
        mi = minimum_mathing_distance(sample_data[sb], ref_data[sb], pop_size, normalize=True, use_EMD=emd)
        scores.append(mi[0])
    scores = np.array(scores)
    return scores

class Evaluator():
    def __init__(self, class_name, voxel_res=28, jsd_in_sphere=True, norm=identity):
        self.class_name = class_name
        self.normalizer = norm
        self.voxel_res = voxel_res
        self.jsd_in_sphere = jsd_in_sphere
        self.splits = ['train', 'test', 'val']
        
    def load_gt_data(self):
        # Load Ground-Truth Data.
        top_gt_dir = '/orions4-zfs/projects/optas/DATA/OUT/iclr/evaluations/gt_data/'
        gt_data = {}
        for s in self.splits:
            gt_file = osp.join(top_gt_dir, self.class_name + '_'+ s +'.npz')
            gt_load = np.load(gt_file)
            gt_data[s] = self.normalizer(gt_load[gt_load.keys()[0]])
            
        self.gt_data = gt_data
    
    def prepare_gt_grid_variables(self):
        # Prepare Ground-Truth grid-variables for JSD.
        gt_grid_vars = {}
        for s in self.splits:
            gt_grid_vars[s] = entropy_of_occupancy_grid(self.gt_data[s], self.voxel_res, self.jsd_in_sphere)[1]
        self.gt_grid_vars = gt_grid_vars

    def prepare_sample_data(self, sample_file, random_seed=42):
        sample_data = {}
        sample_load = np.load(sample_file)      
        sample_data['train'] = self.normalizer(sample_load[sample_load.keys()[0]])

        # Sub-sample train sample data, to form samples for test, val.
        n_train = len(sample_data['train'])
        n_test = len(self.gt_data['test'])
        n_val = len(self.gt_data['val'])
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        test_idx = np.random.choice(np.arange(n_train), n_test, replace=False) 
        val_idx = np.random.choice(np.arange(n_train), n_val, replace=False) 
        sample_data['test'] = sample_data['train'][test_idx]
        sample_data['val'] = sample_data['train'][val_idx]
        self.sample_data = sample_data

    def compute_jsd(self, f_out=sys.stdout):        
        for s in self.splits:
            sample_grid_var = entropy_of_occupancy_grid(self.sample_data[s], self.voxel_res, self.jsd_in_sphere)[1]
            jsd_score = jensen_shannon_divergence(sample_grid_var, self.gt_grid_vars[s])
            print(s, jsd_score, file=f_out)
            if f_out != sys.stdout:
                print(s, jsd_score)
            
           
    def compute_mmd(self, loss='emd', sample=True, n_samples=5, pop_size=200, f_out=sys.stdout):
        if loss == 'emd':
            emd = True
        else:
            emd = False
        
        for s in self.splits:
            scores = sampling_mmd(self.sample_data[s], self.gt_data[s], n_samples, pop_size, emd=emd)      
            print(s, np.mean(scores), np.std(scores), file=f_out)
            if f_out != sys.stdout:
                print(s, np.mean(scores), np.std(scores))