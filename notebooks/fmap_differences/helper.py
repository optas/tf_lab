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

def load_pclouds_of_shapes(top_data_dir, members_per_class, normalize=True, n_pc_points=1024):
    in_pcs = osp.join(top_data_dir, '%d_shape_pc_points.mat' % (members_per_class, ))
    in_pcs = loadmat(in_pcs)
    in_pcs = in_pcs['selected_points']
    n_shapes = len(in_pcs)
    res = np.zeros((n_shapes, n_pc_points, 3))
    for i in xrange(n_shapes):
        pc = Point_Cloud(in_pcs[i])
        pc, _ = pc.sample(n_pc_points)
        res[i] = pc.points
        if normalize:
            pc = Point_Cloud(res[i]).center_in_unit_sphere()
            res[i] = pc.points
    return res
    
def load_diff_maps(in_file, zero_thres):    
    in_diffs = hdf5storage.loadmat(in_file)
    n_shapes = len(in_diffs['ucb'])
    diff_dims = in_diffs['ucb'][1][0].shape
    temp = np.zeros(shape=(n_shapes, ) + diff_dims )
    for i in xrange(n_shapes):
        temp[i] = in_diffs['ucb'][i][0]
    in_diffs = temp    
    return in_diffs

def pose_labels_and_original_index(n_shapes, members_per_class, orinal_class_size):
    pose_labels = np.zeros(n_shapes)
    original_idx = np.zeros(n_shapes, dtype=int)
    c = 0
    s = 0
    for i in range(n_shapes):
        if i % members_per_class == 0:
            c += 1
            s = 0
        pose_labels[i] = c
        original_idx[i] = s + ((i / members_per_class) * orinal_class_size)
        s += 1

    pose_labels -= 1
    return pose_labels, original_idx

def load_gt_latent_params(top_data_dir):
    gt_latent_params = osp.join(top_data_dir, 'gt_shape_params.mat')
    gt_latent_params = loadmat(gt_latent_params)
    gt_latent_params = gt_latent_params['parammat'].T
    return gt_latent_params

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