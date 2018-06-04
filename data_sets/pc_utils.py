import numpy as np

def pclouds_with_zero_mean_in_unit_sphere(in_pclouds):
    ''' Zero MEAN + Max_dist = 0.5
    '''
    pclouds = in_pclouds.copy()
    pclouds = pclouds - np.expand_dims(np.mean(pclouds, axis=1), 1)
    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    dist = np.expand_dims(np.expand_dims(dist, 1), 2)
    pclouds = pclouds / (dist * 2.0)
    return pclouds