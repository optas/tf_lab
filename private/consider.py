'''
Created on Feb 8, 2017

@author: optas
'''



# Put pretrained PCA layer
from sklearn.decomposition import PCA
pca = PCA(n_components=11)
lcodes_in_pc = pca.fit_transform(train_lcodes)
print np.sum(pca.explained_variance_ratio_)
train_lcodes_backfrom_pca = pca.inverse_transform(lcodes_in_pc)
# post_pca = tf.matmul(tf.constant(pca.components_.T), ae.z - tf.constant(pca.mean_, tf.float32))
post_pca = tf.matmul(ae.z - tf.constant(pca.mean_, tf.float32), tf.constant(pca.components_.T))



# Some cool stuff
with tf.control_dependencies([ae.optimizer]):
    loss_after_optimizer = tf.identity(ae.loss)


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


def density_weights(points, radius):
    nn = NearestNeighbors().fit(points)
    indices = nn.radius_neighbors(radius=radius, return_distance=False)
    density = np.array([i.size for i in indices], dtype=np.float128)
    return density

def inverse_normalize_weights(weights):
    w = weights
    inv_w = 1.0 / w[w != 0]
    w[w != 0] = inv_w
    w /= np.sum(w)
    return w.astype(np.float32)

w = inverse_normalize_weights(density_weights(points, radius = 0.1))



import sklearn
from scipy.sparse.csgraph import minimum_spanning_tree as mst

def make_normals_consistent(points, normals, n_neighbors=5):
    ''' It makes the output of MCubes worse... 
    '''
    g = sklearn.neighbors.kneighbors_graph(points, mode='distance', n_neighbors=5)
    g = mst(g, overwrite=True)
    edges = g.nonzero()
    visited = np.zeros(len(points), dtype=np.bool)
    visited[edges[0]] = 1
    for i, j in zip(edges[0], edges[1]):
        if visited[i] and not visited[j]:
            if normals[i].dot(normals[j]) < 0:
                normals[j] = - normals[j]

        elif visited[j] and not visited[i]:
            if normals[j].dot(normals[i]) < 0:
                normals[i] = - normals[i]

        elif not visited[j] and not visited[i]:
            assert False

        elif visited[i] and visited[j]:
            continue
        else:
            assert False

        visited[i] = True
        visited[j] = True
    assert(np.all(visited))
    return normals

