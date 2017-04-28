'''
Created on Feb 8, 2017

@author: optas
'''


# TRYING TO LOAD VERTEX QUALITY FROM MESHLAB

from geo_tool.external_code.python_plyfile.plyfile import PlyData
from geo_tool import Mesh
import numpy as np

test_file = '/Users/optas/Desktop/horse-gallop-21.off.ply'
file_name =test_file
def load_ply(file_name, with_faces=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    p_quality = points['quality']
    points = [np.vstack([points['x'], points['y'], points['z']]).T, p_quality]
    
    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        return points, faces
    else:
        return points

# v, t = load_ply(test_file, with_faces=True)
# Mesh(vertices=v[0], triangles=t).plot(vertex_function = v[1])

def distance_field_from_nn(pointcloud, grid_resolution, k):
    r = grid_resolution
    grid, spacing = compute_3D_grid(r)
    grid = grid.reshape(-1, 3)
    nn = NearestNeighbors(n_neighbors=k).fit(pointcloud)
    distances, _ = nn.kneighbors(grid)
    distances = np.average(distances, axis=1)
    distances = distances.astype(np.float32)
    distances = distances.reshape(r, r, r, 1)
    distances /= spacing
    return distances