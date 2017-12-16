'''
Created on December 14, 2017

@author: optas
'''

from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import warnings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import os.path as osp
from external_tools.binvox_rw.binvox_rw import read_as_3d_array


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


def load_voxel_grids_from_filenames(file_names, n_threads=1, loader=read_shape_net_bin_vox, verbose=False, dtype=np.float32):
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
