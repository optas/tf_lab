import numpy as np
import tensorflow as tf
import os.path as osp

import tf_lab.point_clouds.in_out as pio
from tf_lab.point_clouds.vae import VariationalAutoencoder, Configuration
from tf_lab.point_clouds.in_out import PointCloudDataSet
from tf_lab.point_clouds.in_out import train_validate_test_split
from general_tools.basics import create_dir


DATA_PATH = '/orions4-zfs/projects/lins2/Panos_Space/DATA/ShapeNetPointClouds/from_manifold_meshes/1024/03001627'
file_names = pio.load_filenames_of_input_data(DATA_PATH)
all_pclouds = np.array(pio.load_crude_point_clouds(file_names=file_names)[0])
train_data, val_data, test_data = train_validate_test_split(all_pclouds, train_perc=0.9, validate_perc=0.1, test_perc=0.1, seed=42)
train_data = PointCloudDataSet(train_data)
test_data = PointCloudDataSet(test_data)

n_input = [1024, 3]
n_z = 128
training_epochs = 1000
batch_size = 20
learning_rate = 0.0001
saver_step = 10
loss = 'Bernoulli'

train_dir = '/orions4-zfs/projects/lins2/Panos_Space/DATA/OUT/vae_with_bernoulli'
create_dir(train_dir)

conf = Configuration(n_input, n_z, training_epochs, batch_size, learning_rate=learning_rate,
                     saver_step=saver_step, loss=loss, train_dir=train_dir)

vae = VariationalAutoencoder(conf)
vae.train(train_data, conf)
