import os
import warnings
import numpy as np
import os.path as osp
import tensorflow as tf
from multiprocessing import Pool

from geo_tool import Mesh, Point_Cloud
from geo_tool.in_out.soup import load_crude_point_cloud
from geo_tool.in_out.soup import load_mesh_from_file

from general_tools.rla.three_d_transforms import rand_rotation_matrix
from general_tools.in_out.basics import files_in_subdirs

from .. autopredictors.scripts.helper import points_extension


vscan_search_pattern = '__0__.ply'
blensor_search_pattern = '0_noisy00000.txt'


def _load_crude_pcloud_and_model_id(f_name):
    tokens = f_name.split('/')
    model_id = tokens[-1].split('_')[0]
    class_id = tokens[-2]
    return load_crude_point_cloud(f_name), model_id, class_id


def _load_blensor_incomplete_pcloud(f_name):
    points = load_crude_point_cloud(f_name, permute=[0, 2, 1])
    pc = Point_Cloud(points=points)
    pc.lex_sort()
    pc.center_in_unit_sphere()
    tokens = f_name.split('/')
    return pc.points, tokens[-2], tokens[-3]


def _load_virtual_scan_incomplete_pcloud(f_name, n_samples=1024):
    pc = Point_Cloud(ply_file=f_name)
    pc.permute_points([0, 2, 1])
    pc = pc.sample(n_samples)
    pc.lex_sort()
    pc.center_in_unit_sphere()
    tokens = f_name.split('/')
    model_id = tokens[-1][:-len(vscan_search_pattern)]
    class_id = tokens[-2]
    return pc.points, model_id, class_id


def load_crude_point_clouds(file_names, n_threads=1, loader=_load_crude_pcloud_and_model_id, verbose=False):
    pc = loader(file_names[0])[0]
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print '%d pclouds were loaded. They belong in %d shape-classes.' % (len(pclouds), len(np.unique(class_ids)))

    return pclouds, model_names, class_ids


def load_cc_parts_of_model(model_path):
    raise NotImplementedError()


def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def load_filenames_of_input_data(top_directory, file_extension=points_extension, verbose=False):
    res = []
    for file_name in files_in_subdirs(top_directory, file_extension + '$'):
        res.append(file_name)
    if verbose:
        print '%d files were found.' % (len(res), )
    return res


def load_mesh_filenames(top_directory):
    model_names = []
    for file_name in os.listdir(top_directory):
        model_name = os.path.join(top_directory, file_name, 'models.obj')
        if os.path.exists(model_name):
            model_names.append(model_name)
    return model_names


def chunks(l, n):
    '''Yield successive n-sized chunks from l.
    '''
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def train_validate_test_split(arrays, train_perc=0, validate_perc=0, test_perc=0, shuffle=True, seed=None):
    ''' This is a memory expensive operation since by using slicing it copies the arrays.
    '''

    if not np.allclose((train_perc + test_perc + validate_perc), 1.0):
        assert(False)

    if type(arrays) is not list:
        arrays = [arrays]

    n = arrays[0].shape[0]   # n examples.
    if len(arrays) > 1:
        for a in arrays:
            if a.shape[0] != n:
                assert(False)

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

    if len(train_data) == 1:
        return train_data[0], validate_data[0], test_data[0], r_ind
    else:
        return train_data, validate_data, test_data, r_ind


def shuffle_two_pcloud_datasets(a, b, seed=None):
    n_a = a.num_examples
    n_b = b.num_examples
    frac_a = n_a / (n_a + n_b + 0.0)
    frac_b = n_b / (n_a + n_b + 0.0)

    a = a.point_clouds
    b = b.point_clouds
    joint = np.vstack((a, b))
    _, new_a, new_b = train_validate_test_split([joint], train_perc=0, validate_perc=frac_a, test_perc=frac_b, seed=seed)

    new_a = PointCloudDataSet(new_a)
    new_b = PointCloudDataSet(new_b)
    if (new_a.num_examples != n_a) or (new_b.num_examples != n_b):
        warnings.warn('The size of the resulting datasets have changed (+-1) due to rounding.')

    return new_a, new_b


# TODO -> Make Noise Class 
#             self.noisy_point_clouds = point_clouds.copy()
#             point_range = np.arange(self.n_points)
#             n_distort = int(noise['frac'] * self.n_points)   # How many points will be noised.
#             for i in xrange(self.num_examples):
#                 drop_index = np.random.choice(point_range, n_distort, replace=False)
#                self.noisy_point_clouds[i, drop_index, :] = noise['filler']

def write_model_ids_of_datasets(out_dir, model_ids, r_indices):
    for ind, name in zip(r_indices, ['train', 'val', 'test']):
        with open(osp.join(out_dir, name + '_data.txt'), 'w') as fout:
            for t in model_ids[ind]:
                fout.write(' '.join(t[:]) + '\n')


class PointCloudDataSet(object):
    '''
    See https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py
    '''

    def __init__(self, point_clouds, noise=None, labels=None):
        '''Construct a DataSet.
        Args:
        Output:
            original_pclouds, labels, noise_pclouds
        '''

        self.num_examples = point_clouds.shape[0]

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            self.labels = labels
        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        self.n_points = point_clouds.shape[1]

        if noise is not None:
            assert (type(noise) is np.ndarray)
            self.noisy_point_clouds = noise
#             self.noisy_point_clouds = self.noisy_point_clouds.reshape(self.num_examples, -1)
        else:
            self.noisy_point_clouds = None

#         self.point_clouds = point_clouds.reshape(self.num_examples, -1)
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, seed=None):
        # TODO: The data are copied. Might be unnecessary.
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            if seed is not None:    # Shuffle the data.
                np.random.seed(seed)
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.point_clouds = self.point_clouds[perm]
            self.labels = self.labels[perm]
            if self.noisy_point_clouds is not None:
                self.noisy_point_clouds = self.noisy_point_clouds[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if self.noisy_point_clouds is None:
            return self.point_clouds[start:end], self.labels[start:end], None
        else:
            return self.point_clouds[start:end], self.labels[start:end], self.noisy_point_clouds[start:end]

    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)
        pc = self.point_clouds[perm]
        lb = self.labels[perm]
        ns = None
        if self.noisy_point_clouds is not None:
            ns = self.noisy_point_clouds[perm]
        return pc, lb, ns
