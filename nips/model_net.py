'''
Created on Mar 30, 2017

@author: optas

Associate code and data for manipulating the shaped of Model-Net-10 (and 40).

'''
import re
import os.path as osp
import numpy as np
from geo_tool import Mesh, Point_Cloud
from geo_tool.in_out.soup import load_ply
import geo_tool.solids.mesh_cleaning as cleaning


train_ply_pattern = '(.*)train(.*)\.ply$'
test_ply_pattern = '(.*)test(.*)\.ply$'

net_10_classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

rotation_angles = {'bed': -90, 'desk': -90, 'dresser': 180, 'chair': -90, 'night_stand': 180,     # These aligns model-net-10 with ShapeNetCore.
                   'sofa': 180, 'monitor': 180, 'bathtub': 0, 'table': 0, 'toilet': -90}


def classes_to_integers(net):
    if net == 10:
        classes = sorted(net_10_classes)
        d = {cid: i for i, cid in zip(range(net), classes)}
        return d
    elif net == 40:
        raise NotImplemented()
    else:
        raise ValueError()


def file_to_category(full_file, ending='obj'):
    regex = '([_a-z]+)_[0-9]+\.' + ending + '$'
    regex = re.compile(regex)
    s = regex.search(osp.basename(full_file))
    return s.groups()[0]


def pc_loader(ply_file):
    ending = 'ply'
    drop = len(ending) + 1
    category = file_to_category(ply_file, ending)
    model_id = osp.basename(ply_file)
    model_id = model_id[:-drop]
    return load_ply(ply_file), model_id, category


def pc_sampler(mesh_file, n_samples, save_file=None, dtype=np.float32):
    category = file_to_category(mesh_file)
    rotate_deg = rotation_angles[category]
    in_mesh = Mesh(file_name=mesh_file)
    in_mesh = cleaning.clean_mesh(in_mesh)
    ss_points, _ = in_mesh.sample_faces(n_samples)
    pc = Point_Cloud(points=ss_points.astype(dtype))
    pc.rotate_z_axis_by_degrees(rotate_deg)
    pc.center_in_unit_sphere()
    pc, _ = pc.lex_sort()
    if save_file is not None:
        pc.save_as_ply(save_file)
    return pc
