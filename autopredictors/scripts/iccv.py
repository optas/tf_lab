'''
Created on Mar 22, 2017

@author: optas
'''

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


in_u_sphere_plotting = {'chair': True, 'airplane': True, 'cabinet': True, 'car': True, 'lamp': True, 'sofa': True, 'table': True, 'vessel': True}

azimuth_angles = {'chair': -50, 'airplane': 0, 'cabinet': -40, 'car': -60, 'lamp': 0, 'sofa': -60, 'table': 60, 'vessel': -60}

plotting_color = {'chair': ['green', 'green', 'green'],
                  'airplane': ['blue', 'blue', 'blue'],
                  'cabinet': ['black', 'black', 'black'],
                  'car': ['red', 'red', 'red'],
                  'lamp': ['yellow', 'yellow', 'yellow'],
                  'sofa': ['magenta', 'magenta', 'magenta'],
                  'table': ['cyan', 'cyan', 'cyan'],
                  'vessel': [0, 0.6, 1]}


def plot_3d_point_cloud(pc_cloud, show=True, in_u_sphere=False, axis=None, elev=10, azim=240,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), dot_color=['blue', 'blue', 'blue'],
                        *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    x = pc_cloud.points[:, 0]
    y = pc_cloud.points[:, 1]
    z = pc_cloud.points[:, 2]

    ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, color=dot_color, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)
    plt.axis('off')

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)

    if show:
        plt.show()
    return fig
