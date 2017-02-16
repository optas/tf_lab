import sys
import time
import numpy as np
import os.path as osp
import tensorflow as tf

import tf_lab.point_clouds.in_out as pio
from tf_lab.point_clouds.point_net_ae import PointNetAutoEncoder
from tf_lab.point_clouds.point_net_ae import Configuration as PN_Conf

import tf_lab.models.point_net_based_AE as pnAE
import tf_lab.point_clouds.various_encoders_decoders as enc_dec

from tf_lab.point_clouds.in_out import PointCloudDataSet
from tf_lab.point_clouds.in_out import train_validate_test_split
from tf_lab.point_clouds.in_out import load_crude_point_clouds

from general_tools.in_out import create_dir

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

DATA_PATH = '/orions4-zfs/projects/lins2/Panos_Space/DATA/ShapeNetPointClouds/from_manifold_meshes/1024/03001627/'
TRAIN_DIR = '/orions4-zfs/projects/lins2/Panos_Space/DATA/OUT/simple_ae_with_chairs/'
experiment_name = 'enc_filter_1_complex_decoder_spn'
TRAIN_DIR = osp.join(TRAIN_DIR, experiment_name)
create_dir(TRAIN_DIR)

file_names = pio.load_filenames_of_input_data(DATA_PATH)
all_pclouds, model_names = load_crude_point_clouds(file_names=file_names, n_threads=11)


conf = PN_Conf(n_input = [1024, 3],
               training_epochs = 1000,
               batch_size = 40,
               loss = 'Chamfer',
               train_dir = TRAIN_DIR,
               loss_display_step = 1,
               saver_step = 5,
               learning_rate = 0.00002,
               saver_max_to_keep = 200,
               gauss_augment = {'mu': 0, 'sigma': 0.02},
#               encoder = pnAE.encoder,
               decoder = pnAE.decoder,
               spatial_trans=True
               )



train_data, val_data, test_data = train_validate_test_split(all_pclouds, train_perc=0.8,
                                                            validate_perc=0.1, test_perc=0.1, seed=seed)

train_data = PointCloudDataSet(train_data)
val_data = PointCloudDataSet(val_data)
test_data = PointCloudDataSet(test_data)

ae = PointNetAutoEncoder(experiment_name, conf)
ae.train(train_data, conf)
