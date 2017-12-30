import numpy as np
import hdf5storage
import os.path as osp
import matplotlib.pylab as plt
from scipy.io import loadmat, savemat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.model_selection import train_test_split

from geo_tool import Point_Cloud, Mesh
from tf_lab.voxels.soup import plot_isosurface
from tf_lab.data_sets.numpy_dataset import NumpyDataset

total_shapes = 2000
members_per_pose_class = 250
n_pose_classes = 8

def plot_mesh(in_mesh):
    faces = in_mesh.triangles
    verts = in_mesh.vertices
    verts = Point_Cloud(verts).center_in_unit_sphere().points
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])    
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    
    miv = np.min(verts)
    mav = np.max(verts)
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()
    plt.show()

def sub_collection_indices(sub_size_per_class):
    sub_size = sub_size_per_class * n_pose_classes
    original_idx = np.zeros(sub_size, dtype=int)
    c = 0
    for i in range(0, total_shapes, members_per_pose_class):
        for j in range(sub_size_per_class):
            original_idx[c] = i + j
            c += 1
    return original_idx

def sub_collection_pose_labels(sub_size_per_class):
    n_sub = sub_size_per_class * n_pose_classes
    pose_labels = np.zeros(n_sub)
    c = -1
    for i in range(n_sub):
        if i % sub_size_per_class == 0:
            c += 1
        pose_labels[i] = c    
    return pose_labels


def load_pclouds_of_shapes(top_data_dir, sub_size_per_class, n_pc_points, normalize=False):
    in_pcs = osp.join(top_data_dir, 'uniform_point_clouds_%d_pts.npz' % (n_pc_points, ))
    in_pcs = np.load(in_pcs)
    in_pcs = in_pcs[in_pcs.keys()[0]]
    
    idx = sub_collection_indices(sub_size_per_class)
    in_pcs = in_pcs[idx]
    
    if normalize:
        res = np.zeros_like(in_pcs)
        for i, pts in enumerate(in_pcs):        
            pc = Point_Cloud(pts).center_in_unit_sphere()
            res[i] = pc.points
    else:
        res = in_pcs
    return res
    
    
def load_gt_latent_params(top_data_dir, sub_size_per_class):
    gt_latent_params = osp.join(top_data_dir, 'gt_shape_params.mat')
    gt_latent_params = loadmat(gt_latent_params)
    gt_latent_params = gt_latent_params['parammat'].T
    original_idx = sub_collection_indices(sub_size_per_class)    
    return gt_latent_params[original_idx]

def load_meshes(mesh_ids):
    meshes = []
    for i in mesh_ids:
        in_f = '/orions4-zfs/projects/optas/DATA/Meshes/SCAPE_8_poses_Ananth/Shape%s.off' % (i,)
        in_m = Mesh(file_name=in_f)
        meshes.append(in_m)
    return meshes

def prepare_train_test_val(n_shapes, class_labels, train_per, test_per, seed=None):
    all_ids = np.arange(n_shapes)
    train_ids, rest_ids = train_test_split(all_ids, stratify=class_labels, train_size=train_per, random_state=seed)
    test_ids, val_ids = train_test_split(rest_ids, stratify=class_labels[rest_ids],
                                         train_size=int(n_shapes*test_per), random_state=seed)
    in_data = dict()
    in_data['train'] = train_ids
    in_data['test'] = test_ids
    in_data['val'] = val_ids
    
    return in_data
    
def make_data(in_data, in_feeds, class_labels):
    res = dict()
    for s in ['train', 'test', 'val']:
        idx = in_data[s].copy()
        res[s] = NumpyDataset([in_feeds[idx], class_labels[idx], idx], ['feed', 'labels', 'ids'])
    return res


### DELETE Below
def load_diff_maps(in_file, zero_thres):    
    in_diffs = hdf5storage.loadmat(in_file)
    n_shapes = len(in_diffs['ucb'])
    diff_dims = in_diffs['ucb'][1][0].shape
    temp = np.zeros(shape=(n_shapes, ) + diff_dims )
    for i in xrange(n_shapes):
        temp[i] = in_diffs['ucb'][i][0]
    in_diffs = temp    
    return in_diffs