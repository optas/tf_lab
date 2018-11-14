'''
Created on Apr 21, 2017

@author: optas
'''

import os.path as osp
import numpy as np
import glob

from general_tools.in_out.basics import create_dir
from geo_tool import Point_Cloud
from geo_tool.in_out.soup import load_crude_point_cloud

# Extensions Eric used.
erics_seg_extension = '.seg'
erics_points_extension = '.pts'

eric_part_pattern = '__?__.ply'    # Function 'equisample_parts_via_bootstrap' saved the parts with this pattern as ending.


all_syn_ids = ['02691156', '02773838', '02954340', '02958343', '03001627', '03261776', '03467517',
               '03624134', '03636649', '03790512', '03797390', '03948459', '04099429', '04225987',
               '04379243']


def _prepare_io(data_top_dir, out_top_dir, synth_id, boot_n):
    points_top_dir = osp.join(data_top_dir, synth_id, 'points')
    segs_top_dir = osp.join(data_top_dir, synth_id, 'expert_verified', 'points_label')
    original_density_dir = osp.join(out_top_dir, synth_id, 'original_density')
    bstrapped_out_dir = osp.join(out_top_dir, str(boot_n) + '_bootstrapped', synth_id)
    create_dir(original_density_dir)
    create_dir(bstrapped_out_dir)
    return segs_top_dir, points_top_dir, original_density_dir, bstrapped_out_dir


def eric_original_density(eric_top_dir, out_top_dir, synth_id, dtype=np.float32, permute=[0, 2, 1]):
    ''' Writes out point clouds with a segmentation mask according to Eric's annotation.
    The point clouds are the original point clouds that Eric sampled, no normalization, neither
    density control is performed'''
    
    points_top_dir = osp.join(eric_top_dir, synth_id, 'points')
    segs_top_dir = osp.join(eric_top_dir, synth_id, 'expert_verified', 'points_label')

    no_part = 0 # a segmentation mask equal to zero, should be ignored.
    
    for file_name in glob.glob(osp.join(segs_top_dir, '*' + erics_seg_extension)):
        model_name = osp.basename(file_name)[:-len(erics_seg_extension)]
        pt_file = osp.join(points_top_dir, model_name + erics_points_extension)
        points = load_crude_point_cloud(pt_file, permute=permute).astype(dtype)
        gt_seg = np.loadtxt(file_name, dtype=dtype)
        if len(gt_seg) != len(points):
            raise ValueError('Segmentation mask has different length than the points.')

        good_points = gt_seg > 0
        points = points[good_points]
        gt_seg = np.expand_dims(gt_seg[good_points], 1)
        out_data = np.hstack((points, gt_seg))
        out_file = osp.join(original_density_dir, model_name)
        np.save(out_file, out_data)    
    

def equisample_parts_via_bootstrap(data_top_dir, out_top_dir, synth_id, n_samples=2048, dtype=np.float32):
    ''' For each object in the synth_id extract the parts and bootstrap them into point-clouds with n_samples each.
    '''
    segs_top_dir, points_top_dir, _, bstrapped_out_dir = _prepare_io(data_top_dir, out_top_dir, synth_id, n_samples)

    for file_name in glob.glob(osp.join(segs_top_dir, '*' + erics_seg_extension)):
        model_name = osp.basename(file_name)[:-len(erics_seg_extension)]
        pt_file = osp.join(points_top_dir, model_name + erics_points_extension)
        points = load_crude_point_cloud(pt_file, permute=[0, 2, 1], dtype=dtype)
        gt_seg = np.loadtxt(file_name, dtype=np.float32)
        seg_ids = np.unique(gt_seg)
        
        if seg_ids[0] == 0:
            seg_ids = seg_ids[1:]   # Zero is not a real segment.

        for seg in seg_ids:
            pc = Point_Cloud(points=points[gt_seg == seg])
            pc, _ = pc.sample(n_samples)
            pc, _ = pc.lex_sort()
            seg_token = '__' + str(seg) + '__'
            out_file = osp.join(bstrapped_out_dir, model_name + seg_token)
            pc.save_as_ply(out_file)


def part_pc_loader(ply_file):
    pc = Point_Cloud(ply_file=ply_file)
    tokens = ply_file.split('/')
    model_id = tokens[-1][:-len(eric_part_pattern)]
    part_id = tokens[-1][-len(eric_part_pattern):-(len('.ply'))]
    syn_id = tokens[-2]
    return pc.points, (model_id, part_id), syn_id
