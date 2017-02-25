'''
Created on Feb 24, 2017

@author: optas
'''

import numpy as np
from .. point_clouds.in_out import PointCloudDataSet
from  .. point_clouds import in_out as pio
import geo_tool.in_out.soup as gio
from geo_tool import Point_Cloud
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt


def merge_val_test_data(val_data, test_data):   # TODO delete
    vt_pclouds = np.vstack((val_data[0], test_data[0]))
    n_val = len(val_data[1])
    n_test = len(test_data[1])
    vt_labels = np.vstack((val_data[1].reshape((n_val, 1)), test_data[1].reshape((n_test, 1))))
    vt_labels = vt_labels.reshape(len(vt_labels), )
    val_test_merged = PointCloudDataSet(vt_pclouds, labels=vt_labels)
    return val_test_merged


def latent_embedding_of_entire_dataset(dataset, model, conf):
    batch_size = conf.batch_size
    feed, labels, _ = dataset.full_epoch_data()
    latent = []
    for b in pio.chunks(feed, batch_size):
        latent.append(model.transform(b.reshape([len(b)] + conf.n_input)))

    latent = np.vstack(latent)
    return feed, latent, labels


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
