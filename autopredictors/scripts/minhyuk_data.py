'''
Created on Mar 13, 2017

@author: optas
'''

import numpy as np
import os.path as osp
from general_tools.in_out.basics import files_in_subdirs
from geo_tool import Mesh, Point_Cloud
from geo_tool.solids import mesh_cleaning as cleaning
from tf_lab.autopredictors.evaluate import accuracy_of_completion
from tf_lab.point_clouds.in_out import PointCloudDataSet

test_categories = ['assembly_airplanes', 'assembly_bicycles', 'assembly_chairs', 'coseg_chairs', 'shapenet_tables']
test_axis_swaps = [[2, 0, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]

top_gt_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/Minhyuk/ground_truth_datasets/'
top_bench_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/Minhyuk/benchmark_results/'


def load_file_names_of_category(category_id):
    category = test_categories[category_id]
    print category
    gt_off_dir = osp.join(top_gt_dir, category, 'off')
    gt_off_files = [f for f in files_in_subdirs(gt_off_dir, 'off$')]
    gt_names = [osp.basename(f)[:-len('.off')] for f in gt_off_files]
    ply_file_prefix = osp.join(top_bench_dir, category, 'output')
    ply_incomplete_files = [osp.join(ply_file_prefix, name, name + '_input.ply') for name in gt_names]
    return ply_incomplete_files, gt_off_files, gt_names


def dataset_of_category(category_id, incomplete_n_samples=2048, complete_n_samples=4096):
    ply_incomplete_files, gt_off_files, gt_names = load_file_names_of_category(category_id)
    swap = test_axis_swaps[category_id]
    bline_acc = []
    n_examples = len(ply_incomplete_files)
    inc_pc_data = np.zeros((n_examples, incomplete_n_samples, 3))
    comp_pc_data = np.zeros((n_examples, complete_n_samples, 3))
    labels_data = np.array([test_categories[category_id] + '.' + name for name in gt_names], dtype=object)

    for i in xrange(n_examples):
        inc_pc = Point_Cloud(ply_file=ply_incomplete_files[i])
        inc_pc.permute_points(swap)
        inc_pc, _ = inc_pc.sample(incomplete_n_samples)
        inc_pc.lex_sort()
        inc_pc_data[i] = inc_pc.points
        in_mesh = Mesh(file_name=gt_off_files[i])
        in_mesh = cleaning.clean_mesh(in_mesh)
        in_mesh.swap_axes_of_vertices_and_triangles(swap)
        comp_pc, _ = in_mesh.sample_faces(complete_n_samples)
        comp_pc = Point_Cloud(points=comp_pc)
        comp_pc.lex_sort()
        comp_pc_data[i] = comp_pc.points
        bline_acc.append(accuracy_of_completion(inc_pc_data[i], comp_pc_data[i], thres=0.02))
    return PointCloudDataSet(comp_pc_data, labels=labels_data, noise=inc_pc_data), bline_acc
