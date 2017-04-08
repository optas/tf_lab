'''
Created on Mar 22, 2017

@author: optas
'''


def plotting_default_params(category):
    kwdict = {}
    kwdict['in_u_sphere'] = in_u_sphere_plotting[category]
    kwdict['azim'] = azimuth_angles[category]
    kwdict['color'] = plotting_color[category]
    return kwdict

in_u_sphere_plotting = {'chair': True, 'airplane': True, 'cabinet': True, 'car': True, 'lamp': True, 'sofa': True, 'table': True, 'vessel': True}

azimuth_angles = {'chair': -50, 'airplane': 0, 'cabinet': -40, 'car': -60, 'lamp': 0, 'sofa': -60, 'table': 60, 'vessel': -60}

plotting_color = {'chair': 'g', 'airplane': 'b', 'cabinet': 'black', 'car': 'r', 'lamp': 'yellow', 'sofa': 'magenta', 'table': 'cyan', 'vessel': [0, 0.6, 1]}
