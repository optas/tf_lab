import tensorflow as tf
import os.path as osp
import glob

from autopredictors.scripts.helper import points_extension
from geo_tool.in_out.soup import load_crude_point_cloud
from geo_tool.in_out import load_mesh_from_file 

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


def _bytes_feature(value):
    '''Wrapper for inserting bytes features into Example proto.'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_mesh_to_example(mesh_file):
    '''Build an Example proto for an example.'''
    mesh_data = load_mesh_from_file(mesh_file)
    vertices = mesh_data[0].tostring()
    triangles = mesh_data[1].tostring()

#         model_name = osp.basename(mesh_file).split('_')[0]   # TODO-In the future we need to store those too
#         synset = osp.split(mesh_file)[0].split(osp.sep)[-1]

    # Construct an Example proto object.
    example = tf.train.Example(\
        # Example contains a Features proto object.
        features=tf.train.Features(\
            # Features contains a map of string to Feature proto objects.
            feature={'vertices': _bytes_feature(vertices),
                     'triangles': _bytes_feature(triangles)
                     }))
    return example


def convert_meshes_to_tfrecord(mesh_files, out_dir, data_name):
    '''Converts the input mesh_files to a tfrecord file.
    '''

    out_file = osp.join(out_dir, data_name + tf_record_extension)
    writer = tf.python_io.TFRecordWriter(out_file)

    for mesh_file in mesh_files:
        example = _convert_mesh_to_example(mesh_file)
        # Use the proto object to serialize the example to a string.
        serialized = example.SerializeToString()
        # Write the serialized object to disk.
        writer.write(serialized)
    writer.close()


def read_and_decode_meshes(filename_queue, n_samples):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={'vertices': tf.FixedLenFeature(shape=[], dtype=tf.string),
                                                 'triangles': tf.FixedLenFeature(shape=[], dtype=tf.string)}
                                       )

    vertices = tf.decode_raw(features['vertices'], tf.float32)
    triangles = tf.decode_raw(features['triangles'], tf.int32)

#     vertices.set_shape([ ?? ])   # LIN?
#     triangles.set_shape([ ?? ])

    # PANOS - Here I will add sampling/rotation/noise of the loaded mesh
    in_data = []
    out_data = []

    return in_data, out_data
