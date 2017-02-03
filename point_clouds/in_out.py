import tensorflow as tf
import numpy as np
import os.path as osp
import glob
import os

from autopredictors.scripts.helper import points_extension
from geo_tool.in_out.soup import load_crude_point_cloud
from geo_tool.in_out.soup import load_mesh_from_file
from geo_tool import Mesh, Point_Cloud
from general_tools.rla.three_d_transforms import rand_rotation_matrix

tf_record_extension = '.tfrecord'


def load_crude_point_clouds(top_directory=None, file_names=None):
    pclouds = []
    model_names = []
    if file_names is None:
        file_names = glob.glob(osp.join(top_directory, '*' + points_extension))

    for file_name in file_names:
        pclouds.append(load_crude_point_cloud(file_name))
        model_name = osp.basename(file_name).split('_')[0]
        model_names.append(model_name)
    return pclouds, model_names


def load_filenames_of_input_data(top_directory):
    res = []
    for file_name in glob.glob(osp.join(top_directory, '*' + points_extension)):
        res.append(file_name)

    print '%d files containing  point clouds were found.' % (len(res), )
    return res


def load_mesh_filenames(top_directory):
    model_names = []
    for file_name in os.listdir(top_directory):
        model_name = os.path.join(top_directory, file_name, 'models.obj')
        if os.path.exists(model_name):
            model_names.append(model_name)
    return model_names


def in_out_placeholders(configuration):
    n = configuration.n_points
    e = configuration.original_embedding
    b = configuration.batch_size
    in_signal = tf.placeholder(dtype=tf.float32, shape=(b, n, e), name='input_pclouds')
    gt_signal = tf.placeholder(dtype=tf.float32, shape=(b, n, e), name='output_pclouds')
    return in_signal, gt_signal


def chunks(l, n):
    '''Yield successive n-sized chunks from l.
    '''
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    '''Wrapper for inserting bytes features into Example proto.'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_mesh_to_example(mesh_file):
    '''Build an Example proto for an example.'''
    print mesh_file
    mesh_data = load_mesh_from_file(mesh_file)
    num_vertices = len(mesh_data[0])
    num_triangles = len(mesh_data[1])
    vertices = mesh_data[0].tostring()
    triangles = mesh_data[1].tostring()

    # Construct an Example proto object.
    example = tf.train.Example(\
        # Example contains a Features proto object.
        features=tf.train.Features(\
            # Features contains a map of string to Feature proto objects.
            feature={'vertices': _bytes_feature(vertices),
                     'num_vertices': _int64_feature(num_vertices),
                     'triangles': _bytes_feature(triangles),
                     'num_triangles': _int64_feature(num_triangles)
                     }))
    return example


def _convert_point_cloud_to_example(point_cloud_file):
    '''Build an Example proto for an example.'''
    pc_raw = load_crude_point_cloud(point_cloud_file).tostring()
    model_name = osp.basename(point_cloud_file).split('_')[0]
    synset = osp.split(point_cloud_file)[0].split(osp.sep)[-1]

    # Construct an Example proto object.
    example = tf.train.Example(\
        # Example contains a Features proto object.
        features=tf.train.Features(\
            # Features contains a map of string to Feature proto objects.
            feature={'pc_raw': _bytes_feature(pc_raw),
                     'model_name': _bytes_feature(model_name),
                     'synset': _bytes_feature(synset)
                     }))
    return example


def convert_data_to_tfrecord(data_files, out_dir, data_name, converter):
    '''Converts the input data_files to a tfrecord file.
    Example: convert_data_to_tfrecord(..., converter=_convert_point_cloud_to_example)
    '''

    out_file = osp.join(out_dir, data_name + tf_record_extension)
    writer = tf.python_io.TFRecordWriter(out_file)

    for dfile in data_files:
        example = converter(dfile)
        # Use the proto object to serialize the example to a string.
        serialized = example.SerializeToString()
        # Write the serialized object to disk.
        writer.write(serialized)
    writer.close()


def train_validate_test_split(arrays, train_perc=0, validate_perc=0.5, test_perc=0.5, shuffle=True, seed=None):
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
    for a in arrays:
        train_data.append(a[perm[:train_end]])
        validate_data.append(a[perm[train_end:validate_end]])
        test_data.append(a[perm[validate_end:]])

    if len(train_data) == 1:
        return train_data[0], validate_data[0], test_data[0]
    else:
        return train_data, validate_data, test_data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_and_decode_point_cloud(filename_queue, points_per_cloud):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={'pcloud_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
                                                 'model_name': tf.VarLenFeature(dtype=tf.string),
                                                 'class_name': tf.VarLenFeature(dtype=tf.string)
                                                 })

    pcloud = tf.decode_raw(features['pcloud_raw'], tf.float32)
    pcloud.set_shape([points_per_cloud])
    return pcloud, features['model_name'], features['class_name']


def input_pcloud_data(filenames, batch_size, points_per_cloud, shuffle=True, num_epochs=None):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filenames], num_epochs=num_epochs, shuffle=shuffle)
        example, model_name, _ = read_and_decode_point_cloud(filename_queue, points_per_cloud)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size

        example_batch = tf.train.shuffle_batch(
            [example, model_name], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return example_batch


class PointCloudDataSet(object):
    '''
    See https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py
    '''

    def __init__(self, point_clouds, noise=None, labels=None):
        '''Construct a DataSet.
        Args:
        noise (Dictionary, optional):   noise['frac'], fraction [0,1] of points that will be distorted in every point-cloud.
                                        noise['filler'], the distortion value.
        '''

        self._num_examples = point_clouds.shape[0]

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            self._labels = labels
        else:
            self._labels = np.ones(self._num_examples)

        self._points_in_pcloud = point_clouds.shape[1]

        if noise is not None:
            self._noisy_point_clouds = point_clouds.copy()
            point_range = np.arange(self._points_in_pcloud)
            n_distort = int(noise['frac'] * self._points_in_pcloud)   # How many points will be noised.
            for i in xrange(self.num_examples):
                drop_index = np.random.choice(point_range, n_distort, replace=False)
                self._noisy_point_clouds[i, drop_index, :] = noise['filler']
            self._noisy_point_clouds = self._noisy_point_clouds.reshape(self._num_examples, -1)
        else:
            self._noisy_point_clouds = None

        self._point_clouds = point_clouds.reshape(self._num_examples, -1)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def point_clouds(self):
        return self.point_clouds

    @property
    def noisy_point_clouds(self):
        return self.noisy_point_clouds

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch.
            self._epochs_completed += 1
            # Shuffle the data.
            if seed is not None:
                np.random.seed(seed)
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._point_clouds = self._point_clouds[perm]
            self._labels = self._labels[perm]
            if self._noisy_point_clouds is not None:
                self._noisy_point_clouds = self._noisy_point_clouds[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._noisy_point_clouds is None:
            return self._point_clouds[start:end], self._labels[start:end], None
        else:
            return self._point_clouds[start:end], self._labels[start:end], self._noisy_point_clouds[start:end]
