'''
Created on Aug 11, 2017

@author: optas
'''


import glob
import os.path as osp
import numpy as np


def read_saved_epochs(saved_dir):
    epochs_saved = []
    files = glob.glob(osp.join(saved_dir, 'models.ckpt-*.index'))
    for f in files:
        epochs_saved.append(int(osp.basename(f)[len('models.ckpt-'):-len('.index')]))
        epochs_saved.sort()
    return epochs_saved


def make_train_validate_test_split(arrays, train_perc=0, validate_perc=0, test_perc=0, shuffle=True, seed=None):
    ''' This is a memory expensive operation since by using slicing it copies the input arrays.
    '''

    if not np.allclose((train_perc + test_perc + validate_perc), 1.0):
        raise ValueError()

    if type(arrays) is not list:
        arrays = [arrays]

    n = arrays[0].shape[0]   # n examples.
    if len(arrays) > 1:
        for a in arrays:
            if a.shape[0] != n:
                raise ValueError('All arrays must have the same number of rows/elements.')

    index = np.arange(n)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(index)
    else:
        perm = np.arange(n)

    train_end = int(train_perc * n)
    validate_end = int(validate_perc * n) + train_end

    train_data = []
    validate_data = []
    test_data = []
    r_ind = (perm[:train_end], perm[train_end:validate_end], perm[validate_end:])

    for a in arrays:
        train_data.append(a[r_ind[0]])
        validate_data.append(a[r_ind[1]])
        test_data.append(a[r_ind[2]])

    if len(train_data) == 1:    # Single array split
        return train_data[0], validate_data[0], test_data[0], r_ind
    else:
        return train_data, validate_data, test_data, r_ind
