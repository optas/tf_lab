'''
Created on Mar 22, 2017

@author: optas
'''

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


def plot_in_u_sphere(category):
    if category in ['chair']:
        return True
    else:
        return False


def plot_3d_point_cloud(pc_cloud, show=True, in_u_sphere=False, axis=None, elev=10, azim=240, marker='.', s=8, alpha=.8, figsize=(5, 5), *args, **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    x = pc_cloud.points[:, 0]
    y = pc_cloud.points[:, 1]
    z = pc_cloud.points[:, 2]

    ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, color=['blue', 'blue', 'blue'], *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)
    plt.axis('off')

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)

    if show:
        plt.show()
    return fig
