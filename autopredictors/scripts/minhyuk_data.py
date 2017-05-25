'''
Created on Mar 13, 2017

Module for manipulation of the point-cloud data used in the project of M. Sung:
    http://cs.stanford.edu/~mhsung/projects/structure-completion


@author: optas
'''

import numpy as np
import os.path as osp
from general_tools.in_out.basics import files_in_subdirs
from geo_tool import Mesh, Point_Cloud
from geo_tool.solids import mesh_cleaning as cleaning
from tf_lab.autopredictors.evaluate import accuracy_of_completion
from tf_lab.point_clouds.in_out import PointCloudDataSet


top_gt_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/Point_Clouds/Partial_PCs/Minhyuk_SigAsia_15/ground_truth_datasets'
top_bench_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/Point_Clouds/Partial_PCs/Minhyuk_SigAsia_15/benchmark_results/'

test_categories = ['assembly_airplanes', 'assembly_bicycles', 'assembly_chairs', 'coseg_chairs', 'shapenet_tables']  # Names of the 5 synthetic classes of objects used for testing the method.
test_axis_swaps = [[2, 0, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]                                            # Not all data classes come with a standard x-y-z system.


def load_file_names_of_category(category_id):
    ''' Will return for a given category of data, all the incomplete file names, with corresponding completions.
    '''
    category = test_categories[category_id]
    print category
    gt_off_dir = osp.join(top_gt_dir, category, 'off')
    gt_off_files = [f for f in files_in_subdirs(gt_off_dir, 'off$')]
    gt_names = [osp.basename(f)[:-len('.off')] for f in gt_off_files]

    manifold_files = [osp.join(top_gt_dir, category, 'manifold', name + '.obj') for name in gt_names]
    ply_file_prefix = osp.join(top_bench_dir, category, 'output')
    ply_incomplete_files = [osp.join(ply_file_prefix, name, name + '_input.ply') for name in gt_names]
    return ply_incomplete_files, gt_off_files, manifold_files, gt_names


def dataset_of_category(category_id, incomplete_n_samples=2048, complete_n_samples=4096, manifold=True):
    ''' TODO: fix randomness.
    '''
    ply_incomplete_files, gt_off_files, manifold_files, gt_names = load_file_names_of_category(category_id)
    swap = test_axis_swaps[category_id]
    bline_acc = []
    n_examples = len(ply_incomplete_files)
    inc_pc_data = np.zeros((n_examples, incomplete_n_samples, 3))
    comp_pc_data = np.zeros((n_examples, complete_n_samples, 3))
    labels_data = np.array([test_categories[category_id] + '.' + name for name in gt_names], dtype=object)

    if manifold:
        gt_files = manifold_files
    else:
        gt_files = gt_off_files

    for i in xrange(n_examples):
        inc_pc = Point_Cloud(ply_file=ply_incomplete_files[i])
        inc_pc.permute_points(swap)
        inc_pc, _ = inc_pc.sample(incomplete_n_samples)
        inc_pc.lex_sort()
        inc_pc_data[i] = inc_pc.points

        in_mesh = Mesh(file_name=gt_files[i])
        in_mesh = cleaning.clean_mesh(in_mesh)
        in_mesh.swap_axes_of_vertices_and_triangles(swap)
        comp_pc, _ = in_mesh.sample_faces(complete_n_samples)
        comp_pc = Point_Cloud(points=comp_pc)
        comp_pc.lex_sort()
        comp_pc_data[i] = comp_pc.points
        bline_acc.append(accuracy_of_completion(inc_pc_data[i], comp_pc_data[i], thres=0.02))

    return PointCloudDataSet(comp_pc_data, labels=labels_data, noise=inc_pc_data), bline_acc


class KinectData(object):
    '''Real Scan (Kinect) Data used by Sung.
    '''

    top_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/Minhyuk/kinect_scan_data/'
    model_names = ['chair001', 'chair002', 'chair003', 'chair006', 'table002', 'table004']
    rotation_angles = np.array([-95, -145, 140, 39, 40, 60])  # Pre-processing for NN (i.e., rotate and swap axis)
    perm = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 0, 2], [1, 0, 2]])
    azimuth_angles = np.array([300, 260, 220, 240, 30, 40])   # Info for Visualization
    in_u_s = [False, False, False, False, True, True]

    @classmethod
    def plot_predictions(cls, nn_model):
        incomplete_n_samples = nn_model.n_input[0]
        for i, name in enumerate(cls.model_names):
            print name
            file_name = osp.join(cls.top_dir, name + '_input.ply')
            pc, _ = Point_Cloud(ply_file=file_name).sample(incomplete_n_samples)
            pc.permute_points(cls.perm[i])
            pc.rotate_z_axis_by_degrees(cls.rotation_angles[i])
            pc.center_axis(0)
            pc.center_axis(1)
            pc.center_axis(2)
            recon = nn_model.reconstruct(pc.points.reshape(1, incomplete_n_samples, 3), compute_loss=False)[0]
            recon = Point_Cloud(points=np.squeeze(recon))
            pc.plot(azim=cls.azimuth_angles[i], s=8, in_u_sphere=cls.in_u_s[i])
            recon.plot(azim=cls.azimuth_angles[i], s=8, in_u_sphere=cls.in_u_s[i])
