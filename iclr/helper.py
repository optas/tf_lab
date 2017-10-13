'''
Created on October 6, 2017

@author: optas

A collection of functions that are being made as we go towards the ICLR deadline. They are for `internal` use only and
should be cleared upon publication.

'''

import os.path as osp
import numpy as np

from .. in_out.basics import Data_Splitter
from .. point_clouds.in_out import load_point_clouds_from_filenames, PointCloudDataSet
from .. data_sets.shape_net import pc_loader as snc_loader
from .. data_sets.shape_net import snc_category_to_synth_id

def load_multiple_version_of_pcs(version, syn_id, n_classes, n_pc_points=2048, random_seed=42):
    top_data_dir = '/orions4-zfs/projects/optas/DATA/'

    if n_classes != 1:
        raise ValueError()

    if version == 'uniform_all':
        versions = ['centered', 'centered_2nd_version', 'centered_3rd_version']

    elif version == 'uniform_one':
        versions = ['centered']

    elif version == 'fps':
        versions = ['fps_sampled_in_u_sphere']

    elif version == 'all':
        versions = ['centered', 'centered_2nd_version', 'centered_3rd_version', 'fps_sampled_in_u_sphere']
    else:
        raise ValueError()

    splits = {'train': None, 'val': None, 'test': None}

    for s in splits.keys():
        print 'Loading %s data.' % (s,)
        s_file = osp.join(top_data_dir, 'Point_Clouds/Shape_Net/Splits/single_class_splits/' + syn_id + '/85_5_10/', s + '.txt')
        print s_file
        for _, v in enumerate(versions):
            top_pclouds_path = osp.join(top_data_dir, 'Point_Clouds/Shape_Net/Core/from_manifold_meshes/', v, str(n_pc_points))
            splitter = Data_Splitter(top_pclouds_path, data_file_ending='.ply', random_seed=random_seed)
            pcs_in_split = splitter.load_splits(s_file)
            pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(pcs_in_split, n_threads=20, loader=snc_loader, verbose=True)
            if splits[s] is None:
                splits[s] = PointCloudDataSet(pclouds, labels=syn_ids + '_' + model_ids)
            else:
                splits[s].merge(PointCloudDataSet(pclouds, labels=syn_ids + '_' + model_ids))
    return splits


def find_best_validation_epoch_from_train_stats(train_stats_file):
    held_out_mark = 'On Held_Out:'
    held_out_loss = []
    held_out_epoch = []
    with open(train_stats_file, 'r') as f_in:
        for line in f_in:
            if line.startswith(held_out_mark):
                line_datum = line.replace(held_out_mark, '')
                tokens = line_datum.split()
                held_out_loss.append(float(tokens[1]))
                held_out_epoch.append(int(tokens[0]))
    held_out_loss = np.array(held_out_loss)
    held_out_epoch = np.array(held_out_epoch)
    best_idx = np.argmin(held_out_loss)
    return held_out_loss[best_idx], held_out_epoch[best_idx]


def achlioptas_five_snc_shape_categories():
    category_names = ['airplane', 'car', 'chair', 'sofa', 'table']
    syn_id_dict = snc_category_to_synth_id()
    return category_names, [syn_id_dict[i] for i in category_names]
