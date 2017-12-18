'''
Created on December 14, 2017

@author: optas
'''
import numpy as np
import warnings
import pickle
import os.path as osp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from external_tools.binvox_rw.binvox_rw import read_as_3d_array
from general_tools.simpletons import invert_dictionary

from .. in_out.basics import Data_Splitter
from .. data_sets.numpy_dataset import NumpyDataset
from .. data_sets.shape_net import snc_category_to_synth_id


def plot_isosurface(voxel_grid, iso_val=0):
    verts, faces, _, _ = measure.marching_cubes(voxel_grid, iso_val)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    d0, d1, d2 = voxel_grid.shape
    ax.set_xlim(0, d0)
    ax.set_ylim(0, d1)
    ax.set_zlim(0, d2)
    plt.tight_layout()
    plt.show()


def read_bin_vox_file(binvox_filename, perm_axis=(0, 1, 2), expand_last_dim=True, dtype=np.float32):
    with open(binvox_filename, 'rb') as f:
        vol = read_as_3d_array(f)

    vol = vol.data.astype(dtype)
    vol = vol.transpose(perm_axis)

    if expand_last_dim:
        vol = np.expand_dims(vol, vol.ndim)

    return vol


def load_voxel_grids_from_filenames(file_names, loader, n_threads=1, verbose=False, dtype=np.float32):
    assert(len(file_names) > 1)
    voxel_grid, _, _ = loader(file_names[0])
    voxel_dims = list(voxel_grid.shape)
    voxel_grids = np.empty([len(file_names)] + voxel_dims, dtype=dtype)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        voxel_grids[i], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(voxel_grids):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} voxel-grids were loaded. They belong in {1} shape-classes.'.format(len(voxel_grids), len(np.unique(class_ids))))

    return voxel_grids, model_names, class_ids


def read_shape_net_bin_vox(binvox_filename):
    tokens = binvox_filename.split('/')
    model_id = tokens[-2]
    class_id = tokens[-3]
    vol_grid = read_bin_vox_file(binvox_filename, perm_axis=(0, 2, 1))
    return vol_grid, model_id, class_id


def read_tartachenko_car_bin_vox(binvox_filename):
    model_id = osp.basename(binvox_filename)
    model_id = model_id.split('.')[0]
    vol_grid = read_bin_vox_file(binvox_filename, perm_axis=(0, 2, 1))
    return vol_grid, model_id, 'car'


def read_phuoc_bin_vox(in_f):
    model_id = osp.basename(in_f)[len('model_normalized_'):-len('_clean.binvox')]
    vol_grid = read_bin_vox_file(in_f, perm_axis=(0, 2, 1))
    return vol_grid, model_id, 'chair'


def load_data_for_rebuttal(load_tartachenko, load_phuoc, class_name, resolution):
    top_data_dir = '/orions4-zfs/projects/optas/DATA'

    in_data = dict()
    if load_tartachenko:
        class_name = 'car'
        top_voxel_dir = osp.join(top_data_dir, 'Voxels/Tartatchenko_Car_Voxels/sn_car_binvox_' + str(resolution))
        splits = ['train', 'test']
        data_in_split = dict()
        data_in_split['train'] = range(6000)
        data_in_split['test'] = range(6000, 7497)
        for s in splits:
            data_in_split_full_path = []
            for d in data_in_split[s]:
                d_file = osp.join(top_voxel_dir, '%04d' % (d,) + '.binvox')
                if osp.exists(d_file):
                    data_in_split_full_path.append(d_file)
            voxel_grids, model_names, class_ids = load_voxel_grids_from_filenames(data_in_split_full_path, loader=read_tartachenko_car_bin_vox,
                                                                                  n_threads=10)
            in_data[s] = NumpyDataset([voxel_grids, class_ids + '_' + model_names], ['voxels', 'labels'])
    else:
        syn_id = snc_category_to_synth_id()[class_name]
        top_voxel_dir = osp.join(top_data_dir, 'Voxels/Choy_ShapeNetVox32')
        if load_phuoc:
            resolution = 64
            loader = read_phuoc_bin_vox
            top_voxel_dir = osp.join(top_data_dir, 'Voxels/Two_Phuong_Chair_64')
            phu_dict_file = '/orions4-zfs/projects/optas/DATA/OUT/2d_to_pc/from_phuoc/phuoc_to_sn_map.pickle'
            with open(phu_dict_file, 'rb') as fin:
                phuoc_to_sn_map = pickle.load(fin)
                sn_id_to_phuoc = invert_dictionary(phuoc_to_sn_map)
        else:
            loader = read_shape_net_bin_vox

        splitter = Data_Splitter(top_voxel_dir, data_file_ending='.binvox')
        splits = ['train', 'test', 'val']
        in_data = dict()

        for s in splits:
            split_file = osp.join(top_data_dir, 'Point_Clouds/Shape_Net/Splits/single_class_splits/' + syn_id + '/85_5_10/', s + '.txt')
            data_in_split = splitter.load_splits(split_file, full_path=False)
            data_in_split_full_path = []
            for d in data_in_split:
                _, model_id = d.split('_')
                if load_phuoc:
                    ph_code = sn_id_to_phuoc[model_id]
                    d_file = osp.join(top_voxel_dir, ph_code + '_clean.binvox')
                else:
                    d_file = osp.join(top_voxel_dir, syn_id, model_id, 'model.binvox')

                if osp.exists(d_file):
                    data_in_split_full_path.append(d_file)
                else:
                    print 'missing:', d_file 

            voxel_grids, model_names, class_ids = load_voxel_grids_from_filenames(data_in_split_full_path, loader=loader,
                                                                                  n_threads=10)
            in_data[s] = NumpyDataset([voxel_grids, class_ids + '_' + model_names], ['voxels', 'labels'])

    return in_data