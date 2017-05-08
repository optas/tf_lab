'''
Created on May 3, 2017

@author: optas
'''

import numpy as np
from geo_tool import Point_Cloud


def compute_3D_grid(resolution=32):
    '''Returns the center coordinates of each cell of a 3D Grid with resolution^3 cells.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in xrange(resolution):
        for j in xrange(resolution):
            for k in xrange(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5
    return grid, spacing


def compute_3D_sphere(resolution=32):
    grid, spacing = compute_3D_grid(resolution=32)
    pc = Point_Cloud(grid.reshape(-1, 3)).center_in_unit_sphere()
    return pc.points, spacing
