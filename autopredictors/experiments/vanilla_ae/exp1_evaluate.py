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
from autopredictors.tf_experiments.evaluate.basics import generalization_error
from autopredictors.tf_experiments.plotting.basics import plot_train_val_test_curves, plot_reconstructions_at_epoch

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


DATA_PATH = '/orions4-zfs/projects/lins2/Panos_Space/DATA/ShapeNetPointClouds/from_manifold_meshes/1024/03001627/'
TRAIN_DIR = '/orions4-zfs/projects/lins2/Panos_Space/DATA/OUT/simple_ae_with_chairs/'
experiment_name = 'enc_filter_1_complex_decoder'
TRAIN_DIR = osp.join(TRAIN_DIR, experiment_name)
create_dir(TRAIN_DIR)

file_names = pio.load_filenames_of_input_data(DATA_PATH)
all_pclouds, model_names = load_crude_point_clouds(file_names=file_names, n_threads=11)
train_data, val_data, test_data = train_validate_test_split(all_pclouds, train_perc=0.8,
                                                            validate_perc=0.1, test_perc=0.1, seed=seed)

train_data = PointCloudDataSet(train_data)
val_data = PointCloudDataSet(val_data)
test_data = PointCloudDataSet(test_data)


conf = PN_Conf(n_input = [1024, 3],
               training_epochs = 1000,
               batch_size = 40,
               loss = 'Chamfer',
               train_dir = TRAIN_DIR,
               loss_display_step = 1,
               saver_step = 5,
               learning_rate = 0.0001,
               saver_max_to_keep = 200,
               gauss_augment = {'mu': 0,'sigma': 0.02},
               encoder = pnAE.encoder,
               decoder = pnAE.decoder
               )


ae = PointNetAutoEncoder(experiment_name, conf)

### Evaluation

SAVEDIR = osp.join('/orions4-zfs/projects/lins2/Panos_Space/DATA/OUT/model_evaluation/vanilla_ae', experiment_name)
create_dir(SAVEDIR)
gen_error, best_epoch, stats = generalization_error(ae, train_data, test_data, val_data, conf)
print 'Best generalization error %f (at epoch %d)' % (gen_error, best_epoch)
np.savetxt(osp.join(SAVEDIR, 'error-stats.txt'), stats)
plot_train_val_test_curves(stats, SAVEDIR, best_epoch, show=False)

plot_reconstructions_at_epoch(best_epoch, ae, test_data, conf, save_dir=osp.join(SAVEDIR, 'test_images_epoch_' + str(best_epoch)))
plot_reconstructions_at_epoch(best_epoch, ae, train_data, conf, save_dir=osp.join(SAVEDIR, 'train_images_epoch_' + str(best_epoch)), max_plot=100)

last_epoch = stats[-1, 0]
plot_reconstructions_at_epoch(last_epoch, ae, test_data, conf, save_dir=osp.join(SAVEDIR, 'test_images_epoch' + str(last_epoch)))  # Also print models at last epoch.

