
# coding: utf-8

# In[2]:

# Stop warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[3]:

import os.path as osp
import numpy as np
from tf_lab.nips.evaluate_gan import entropy_of_occupancy_grid, jensen_shannon_divergence
from tf_lab.nips.helper import pclouds_centered_and_half_sphere
import argparse


# In[43]:

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', type=str, default = '', help='Directory of point-cloud samples.', required=True)
parser.add_argument('--ref', type=str, default = '', help='Path to reference point-cloud.', required=True)
parser.add_argument('--out_file', type=str, help='Save results in this file.', required=True)
parser.add_argument('--epochs', type=list, default = [1, 3, 10, 30, 100, 300, 400, 500], help='Epochs to evaluate.')
opt = parser.parse_args()


# In[46]:

n_pc_samples = 2048
cmp_in_sphere = True
voxel_resolution = 28


# In[47]:

gt_data = np.load(opt.ref)  # Load Ground-Truth Data.
gt_data = gt_data[gt_data.keys()[0]]
gt_data = pclouds_centered_and_half_sphere(gt_data)
_, gt_grid_var = entropy_of_occupancy_grid(gt_data, voxel_resolution, in_sphere=cmp_in_sphere)


# In[1]:

buf_size = 1 # flush each line
fout = open(opt.out_file, 'a', buf_size)
fout.write('Metric Epoch Measurement\n')
print 'Saving measurements at: ' + opt.out_file
for epoch in opt.epochs: 
    sample_file = osp.join(opt.sample_dir, 'epoch_%d.npz' % (epoch,) )
    sample_data = np.load(sample_file)
    sample_data = sample_data[sample_data.keys()[0]]
    sample_data = pclouds_centered_and_half_sphere(sample_data)
    _, sample_grid_var = entropy_of_occupancy_grid(sample_data, voxel_resolution, in_sphere=cmp_in_sphere)
    jsd_epoch = jensen_shannon_divergence(sample_grid_var, gt_grid_var)
    log_data = 'JSD %d %f' % (epoch, jsd_epoch)
    print log_data
    fout.write(log_data + '\n')
fout.close()

