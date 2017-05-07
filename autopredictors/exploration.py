'''
Created on February 24, 2017

@author: optas
'''

import numpy as np
import warnings
try:
    from sklearn.neighbors import NearestNeighbors
except:
    warnings.warn('Sklearn library is not installed.')

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt

import geo_tool.in_out.soup as gio
from geo_tool import Point_Cloud

from .. point_clouds.in_out import apply_augmentations, PointCloudDataSet
from .. point_clouds import in_out as pio

from general_tools.simpletons import iterate_in_chunks


def latent_embedding_of_entire_dataset(dataset, model, conf, feed_original=True, apply_augmentation=False):
    '''
    Observation: the NN-neighborhoods seem more reasonable when we do not apply the augmentation.
    '''
    batch_size = conf.batch_size
    original, ids, noise = dataset.full_epoch_data(shuffle=False)

    if feed_original:
        feed = original
    else:
        feed = noise
        if feed is None:
            feed = original

    feed_data = feed
    if apply_augmentation:
        feed_data = apply_augmentations(feed, conf)

    latent = []
    for b in iterate_in_chunks(feed_data, batch_size):
        latent.append(model.transform(b.reshape([len(b)] + conf.n_input)))

    latent = np.vstack(latent)
    return feed, latent, ids

def embedding_of_entire_dataset_at_tensor(dataset, model, conf, tensor_name, feed_original=True,
                                            apply_augmentation=False):
    '''
    Observation: the next layer after latent (z) might be something interesting.
    tensor_name: e.g. model.name + '_1/decoder_fc_0/BiasAdd:0'
    '''
    batch_size = conf.batch_size
    original, ids, noise = dataset.full_epoch_data(shuffle=False)

    if feed_original:
        feed = original
    else:
        feed = noise
        if feed is None:
            feed = original

    feed_data = feed
    if apply_augmentation:
        feed_data = apply_augmentations(feed, conf)

    latent = []
    latent_tensor = model.graph.get_tensor_by_name(tensor_name)
    for b in iterate_in_chunks(feed_data, batch_size):
        toappend = model.sess.run(latent_tensor,
                                  feed_dict={model.x : b.reshape([len(b)]+conf.n_input)})
        latent.append(toappend)
    latent = np.vstack(latent)
    return feed, latent, ids

def load_pcloud_with_segmentation(pts_file, seg_file, n_samples):
    '''Specific to Eric' data.
    '''
    seg = gio.load_annotation_of_points(seg_file)
    pts = gio.load_crude_point_cloud(pts_file)
    tokens = pts_file.split('/')
    syn_id = tokens[-3]
    model_id = tokens[-1][:-4]
    assert(len(seg) == len(pts))
    pc = Point_Cloud(points=pts)
    pc.permute_points([0, 2, 1])
    pc, s_order = pc.sample(n_samples)
    seg = seg[s_order]
    pc, lex_order = pc.lex_sort()
    seg = seg[lex_order]
    pc.center_in_unit_sphere()
    return pc.points, seg, syn_id, model_id


def plot_mesh_2(in_mesh, show=True, in_u_sphere=False):
    '''Alternative to plotting a mesh with matplotlib.
    Need to find way to colorize vertex/faces.'''

    fig = plt.figure()
    ax = Axes3D(fig)
    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    v = in_mesh.vertices
    tri = Poly3DCollection(v[in_mesh.triangles], facecolors='g', edgecolor='g')
    ax.add_collection3d(tri)
    if show:
        plt.show()
    else:
        return fig


def find_neighbors(X, Y=None, k=10):
    '''If Y is not provided, it returns the k neighbors of each point in the X dataset. Otherwise, it returns the
    k neighbors of X in Y.
    '''
    s = 0
    if Y is None:
        Y = X
        k = k + 1   # First neighbor is one's shelf.
        s = 1       # Used to drop the first-returned neighbor if needed.

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nn.kneighbors(Y)
    distances = distances[:, s:]
    indices = indices[:, s:]
    return indices, distances


def write_out_neighborhoods(write_out_file, X_labels, neighbors, distances, Y_labels=None):
    '''
    write_out_neighborhoods('train_10_neighbs.txt', tr_labels, neighbors, distances)
    '''

    if Y_labels is None:
        Y_labels = X_labels

    with open(write_out_file, 'w') as fout:
        for i, l, d in zip(range(len(Y_labels)), X_labels[neighbors], distances):
            ls = ', '.join(l) + '\n'
            ds = ', '.join(d.astype('str')) + '\n'
            fout.write(Y_labels[i] + '\n' + ls + ds)
