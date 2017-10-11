
# coding: utf-8

# In[2]:

import warnings
import argparse
import os.path as osp
import numpy as np

from pcloud_benchmark.evaluate_gan import minimum_mathing_distance
from tf_lab.nips.helper import pclouds_centered_and_half_sphere


# In[4]:

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', type=str, default = '', help='Directory of point-cloud samples.', required=True)
parser.add_argument('--ref', type=str, default = '', help='Path to reference point-cloud.', required=True)
parser.add_argument('--out_file', type=str, help='Save results in this file.', required=True)
parser.add_argument('--epochs', type=list, default = [1, 3, 10, 30, 100, 300, 400, 500], help='Epochs to evaluate.')
opt = parser.parse_args()


# In[4]:

# class Empty():
#     def __init__(self):
#         pass
# opt = Empty()
# opt.sample_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/OUT/icml/synthetic_point_clouds/nb_gan_ae_14_emd_chair_2048_best_epoch'
# opt.ref = '/orions4-zfs/projects/lins2/Panos_Space/DATA/OUT/nips/our_synthetic_samples/ground_truth/gt_all_chair.npz'
# opt.out_file = '/orions4-zfs/projects/lins2/Panos_Space/DATA/OUT/icml/synthetic_point_clouds/nb_gan_ae_14_emd_chair_2048_best_epoch/mmd_stats.txt'
# opt.epochs = [1, 3, 10, 30, 100, 300, 400, 500]


# In[5]:

n_pc_samples = 2048
batch_size = 1000
reduce_gt = 1000      # (Or None)
reduce_samples = 1000 # (Or None)


# In[7]:

gt_data = np.load(opt.ref)  # Load Ground-Truth Data.
gt_data = gt_data[gt_data.keys()[0]]
gt_data = pclouds_centered_and_half_sphere(gt_data)

if reduce_gt is not None:
    gt_data = gt_data[:reduce_gt, :]


# In[5]:

buf_size = 1 # flush each line
fout = open(opt.out_file, 'w', buf_size)
fout.write('#Metric Epoch Measurement\n')
print 'Saving measurements at: ' + opt.out_file

for epoch in opt.epochs:
    sample_file = osp.join(opt.sample_dir, 'epoch_%d.npz' % (epoch,) )
    sample_data = np.load(sample_file)
    sample_data = sample_data[sample_data.keys()[0]]
    sample_data = pclouds_centered_and_half_sphere(sample_data)    
    if reduce_samples is not None:
        sample_data = sample_data[:reduce_samples, :]
    mmd_epoch = minimum_mathing_distance(sample_data, gt_data, batch_size)[0]
    log_data = 'MMD %d %f' % (epoch, mmd_epoch)    
    print log_data
    fout.write(log_data + '\n')
fout.close()

