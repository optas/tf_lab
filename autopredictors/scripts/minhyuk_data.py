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


top_gt_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/Point_Clouds/Partial_PCs/Minhyuk_SigAsia_15/ground_truth_datasets'   # Complete data top dir.
top_bench_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/Point_Clouds/Partial_PCs/Minhyuk_SigAsia_15/benchmark_results/'   # Incomplete data top-dir.

test_categories = ['assembly_airplanes', 'assembly_bicycles', 'assembly_chairs', 'coseg_chairs', 'shapenet_tables']  # Names of the 5 synthetic classes of objects used for testing the method.

# Not all data classes come with a standard x-y-z system.
test_axis_swaps = {'assembly_chairs': [2, 0, 1],
                   'assembly_airplanes': [0, 1, 2],
                   'coseg_chairs': [0, 1, 2],
                   'shapenet_tables': [0, 1, 2]
                   }

rotation_categories = {'assembly_chairs': [0] * 64,
                       'assembly_airplanes': [0] * 58,
                       'coseg_chairs': [-90] * 291,
                       'shapenet_tables': [0] * 37
                       }

shrink_categories = {'assembly_chairs': [1] * 64,
                     'assembly_airplanes': [1] * 58,
                     'coseg_chairs': [1] * 291,
                     'shapenet_tables': [1] * 37
                     }


def bounding_box_of_3d_points(points):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])
    return xmin, ymin, zmin


def load_file_names_of_category(category_name):
    ''' Will return for a given category of data, all the incomplete file names, with corresponding completions.
    '''

    gt_off_dir = osp.join(top_gt_dir, category_name, 'off')
    gt_off_files = [f for f in files_in_subdirs(gt_off_dir, 'off$')]
    gt_names = [osp.basename(f)[:-len('.off')] for f in gt_off_files]

    manifold_files = [osp.join(top_gt_dir, category_name, 'manifold', name + '.obj') for name in gt_names]
    ply_file_prefix = osp.join(top_bench_dir, category_name, 'output')
    ply_incomplete_files = [osp.join(ply_file_prefix, name, name + '_input.ply') for name in gt_names]
    ply_minhyuk_files = [osp.join(ply_file_prefix, name, name + '_0_fusion.ply') for name in gt_names]
    return ply_incomplete_files, ply_minhyuk_files, gt_off_files, manifold_files, gt_names


def minhyuk_completions(category_name, n_samples=4096):
    '''Returns a point-cloud with n_samples points, that was sub-sampled from the \'completed\' point-cloud Sung's method created.
    '''
    _, ply_minhyuk_files, _, _, _ = load_file_names_of_category(category_name)
    swap = test_axis_swaps[category_name]
    n_examples = len(ply_minhyuk_files)
    minhyuk_pc_data = np.zeros((n_examples, n_samples, 3))
    np.random.seed(42)  # TODO Is this enough? Push inside for loop? Need to save at disk?
    for i in xrange(n_examples):
        minhyuk_pc = Point_Cloud(ply_file=ply_minhyuk_files[i])
        minhyuk_pc.permute_points(swap)
        minhyuk_pc, _ = minhyuk_pc.sample(n_samples)
        minhyuk_pc.lex_sort()
        minhyuk_pc_data[i] = minhyuk_pc.points
    return minhyuk_pc_data


def groundtruth_point_clouds(category_name, n_samples):
    _, _, gt_off_files, _, gt_names = load_file_names_of_category(category_name)
    swap = test_axis_swaps[category_name]
    n_examples = len(gt_off_files)
    comp_pc_data = np.zeros((n_examples, n_samples, 3))
    np.random.seed(42)
    for i in xrange(n_examples):
        in_mesh = Mesh(file_name=gt_off_files[i])
        in_mesh = cleaning.clean_mesh(in_mesh)
        in_mesh.swap_axes_of_vertices_and_triangles(swap)
        comp_pc, _ = in_mesh.sample_faces(n_samples)
        comp_pc = Point_Cloud(points=comp_pc)
        comp_pc.lex_sort()
        comp_pc_data[i] = comp_pc.points
    return comp_pc_data, gt_names


def dataset_of_category(category_name, incomplete_n_samples=2048, complete_n_samples=4096, manifold=True):
    ''' TODO: fix randomness.
    '''
    ply_incomplete_files, _, gt_off_files, manifold_files, gt_names = load_file_names_of_category(category_name)
    swap = test_axis_swaps[category_name]
    bline_acc = []
    n_examples = len(ply_incomplete_files)
    inc_pc_data = np.zeros((n_examples, incomplete_n_samples, 3))
    comp_pc_data = np.zeros((n_examples, complete_n_samples, 3))
    labels_data = np.array([category_name + '.' + name for name in gt_names], dtype=object)

    if manifold:
        gt_files = manifold_files
    else:
        gt_files = gt_off_files

    np.random.seed(42)

    for i in xrange(n_examples):
        inc_pc = Point_Cloud(ply_file=ply_incomplete_files[i])
        inc_pc.permute_points(swap)
        inc_pc, _ = inc_pc.sample(incomplete_n_samples)
        inc_pc.lex_sort()

        in_mesh = Mesh(file_name=gt_files[i])
        in_mesh = cleaning.clean_mesh(in_mesh)
        in_mesh.swap_axes_of_vertices_and_triangles(swap)
        comp_pc, _ = in_mesh.sample_faces(complete_n_samples)
        comp_pc = Point_Cloud(points=comp_pc)
        comp_pc.lex_sort()

        # Lin
        gap = comp_pc.center_axis()[1]
        inc_pc.points = inc_pc.points - gap
        inc_pc.rotate_z_axis_by_degrees(rotation_categories[category_name][i])
        comp_pc.rotate_z_axis_by_degrees(rotation_categories[category_name][i])
        xmin, ymin, zmin = bounding_box_of_3d_points(comp_pc.points)
        shrink_categories[category_name][i] = np.sqrt(xmin ** 2 + ymin ** 2 + zmin ** 2) * 2
        inc_pc_data[i] = inc_pc.points / (shrink_categories[category_name][i])
        comp_pc_data[i] = comp_pc.points / (shrink_categories[category_name][i])
        # end Lin

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
