import os.path as osp
import numpy as np

from mayavi import mlab as mayalab
from general_tools.in_out.basics import read_header_of_np_saved_txt
from geo_tool import Point_Cloud
from geo_tool.in_out.soup import load_crude_point_cloud 


def load_grad_data(grad_file, original_pcloud_folder=None):
    header = read_header_of_np_saved_txt(grad_file).splitlines()
    model_id = header[0].split('=')[1]
    bottle_n = int(header[1].split('=')[1])
    grad = np.loadtxt(grad_file)
    model_data = dict()
    model_data[model_id] = dict()
    model_data[model_id]['grad_x'] = grad[:, ::3]
    model_data[model_id]['grad_y'] = grad[:, 1::3]
    model_data[model_id]['grad_z'] = grad[:, 2::3]    
    return model_data


def plot_vector_field(points, vx, vy, vz):
    mayalab.quiver3d(points[:, 0], points[:, 1], points[:, 2], vx, vy, vz)
    mayalab.show()
    

def combine_grad_from_all_hidden(model_data, norm=True):    
    def grad_single_coord(k):    
        g = model_data[k]
        t = np.sum(g, 1)
        if norm:
            t /= g.shape[1]
        return t
    
    keys = ['grad_x', 'grad_y', 'grad_z']
    res = []
    for k in keys:
        res.append(grad_single_coord(k))
        
    return res[0], res[1], res[2]


def grad_norms(model_data):
    def grad_single_coord(k):    
        return l2_norm(model_data[k], axis=1)

    keys = ['grad_x', 'grad_y', 'grad_z']
    res = []
    for k in keys:
        res.append(grad_single_coord(k))
        
    return res[0], res[1], res[2]
