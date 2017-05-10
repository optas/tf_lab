'''
Created on May 9, 2017

@author: optas
'''

import matplotlib.pyplot as plt
from geo_tool import Point_Cloud


def plot_interpolations(inter_clouds, grid_size, fig_size=(50, 50)):
    fig = plt.figure(figsize=fig_size)
    c = 1
    for cloud in inter_clouds:
        plt.subplot(grid_size[0], grid_size[1], c, projection='3d')
        plt.axis('off')
        ax = fig.axes[c - 1]
        Point_Cloud(points=cloud).plot(axis=ax, show=False)
        c += 1
    return fig