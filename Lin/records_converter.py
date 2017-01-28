# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts raw point data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from global_variables import *
from load_data import load_data

tf.app.flags.DEFINE_string('directory', '/orions4-zfs/projects/lins2/Lin_Space/DATA/Lin_Data/TFRecords/','Directory to write the converted result')
FLAGS = tf.app.flags.FLAGS

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(raw_point_clouds,model_list, name):
  num_examples = len(raw_point_clouds)

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    pc_raw = raw_point_clouds[index].tostring()
    model_id = model_list[index]
    example = tf.train.Example(features=tf.train.Features(feature={
        'pc_raw': _bytes_feature(pc_raw),
        'model_id':_bytes_feature(model_id)}))
    writer.write(example.SerializeToString())
  writer.close()

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'pc_raw':tf.FixedLenFeature([],tf.string),
                'model_id':tf.VarLenFeature(tf.string),
            })

    point_cloud = tf.decode_raw(features['pc_raw'],tf.float32)
    point_cloud.set_shape([Npoint*3])
    model_id = features['model_id']
    return point_cloud,model_id


def main(argv):
  # Get the data.
  train_data_sets,train_model_list = load_data(train_file_list)
  test_data_sets,test_model_list = load_data(test_file_list)

  # Converts to TFRecords
  convert_to(train_data_sets,train_model_list, 'chair_train')
  convert_to(test_data_sets,test_model_list,'chair_test')

if __name__ == '__main__':
  tf.app.run()
