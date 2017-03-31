'''
Created on Mar 30, 2017

@author: optas
'''
import re
import os.path as osp
import numpy as np
from geo_tool import Mesh, Point_Cloud
import geo_tool.solids.mesh_cleaning as cleaning


rotation_angles = {'bed': -90, 'desk': -90, 'dresser': 180, 'chair': -90, 'night_stand': 180,     # These aligns model-net-10 with ShapeNetCore.
                   'sofa': 180, 'monitor': 180, 'bathtub': 0, 'table': 0, 'toilet': -90}


def file_to_category(full_file):
    regex = '([_a-z]+)_[0-9]+\.obj$'
    regex = re.compile(regex)
    s = regex.search(osp.basename(full_file))
    return s.groups()[0]


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
