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
        model_name = os.path.join(top_directory,file_name,'model.obj')
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
#         model_name = osp.basename(mesh_file).split('_')[0]   # TODO-In the future we need to store those too
#         synset = osp.split(mesh_file)[0].split(osp.sep)[-1]

    # Construct an Example proto object.
    example = tf.train.Example(\
        # Example contains a Features proto object.
        features=tf.train.Features(\
            # Features contains a map of string to Feature proto objects.
            feature={'vertices': _bytes_feature(vertices),
                     'num_vertices': _int64_feature(num_vertices)
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
                                       features={'vertices': tf.VarLenFeature(dtype=tf.float32),
                                                 'num_vertices': tf.FixedLenFeature([],dtype=tf.int64)}
                                       )

    print features['vertices'].get_shape()
  #  triangles = tf.decode_raw(features['triangles'], tf.int32)

#     vertices.set_shape([ ?? ])   # LIN?
#     triangles.set_shape([ ?? ])

#    in_mesh = Mesh(vertices=vertices, triangles=triangles)
#    points, _ = in_mesh.sample_faces(n_samples)
#    points = points * rand_rotation_matrix()   # Keep If you want to apply a random rotation (don't forget to git-pull the 'general-tools')

    # Adding Gaussian noise.
#    mu = 0
#    sigma = 0.01  # LIN How much should the sigma be?
#    gnoise = np.random.normal(mu, sigma, points.shape[0])
#    gnoise = np.tile(gnoise, (3, 1)).T
#    points += gnoise

    # Center in Unit Sphere and Lex-Sort
#    pc = Point_Cloud(points=points)
#    pc.center_in_unit_sphere()
#    pc, lex_index = pc.lex_sort()

#    labels = points
    return features['vertices']

if __name__ == '__main__':
    mesh_files = load_mesh_filenames('/orions4-zfs/projects/lins2/Panos_Space/DATA/ShapeNetManifold/20000/03001627/')
    print mesh_files
    convert_meshes_to_tfrecord(mesh_files, '/orions4-zfs/projects/lins2/Lin_Space/DATA/Lin_Data/mesh/','chair')
