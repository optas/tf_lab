'''
Created on Feb 24, 2017

@author: optas
'''
import tensorflow as tf

tf_record_extension = '.tfrecord'


# TODO -> Make Noise Class 
#             self.noisy_point_clouds = point_clouds.copy()
#             point_range = np.arange(self.n_points)
#             n_distort = int(noise['frac'] * self.n_points)   # How many points will be noised.
#             for i in xrange(self.num_examples):
#                 drop_index = np.random.choice(point_range, n_distort, replace=False)
#                self.noisy_point_clouds[i, drop_index, :] = noise['filler']

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


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot