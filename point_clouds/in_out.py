import tensorflow as tf
import os.path as osp
import glob

from autopredictors.scripts.helper import points_extension
from geo_tool.in_out.soup import load_crude_point_cloud


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
