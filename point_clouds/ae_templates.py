'''
Created on September 2, 2017

@author: optas
'''
import numpy as np

from . encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only


def conv_architecture_ala_nips_17(n_pc_points):

    if n_pc_points == 2048:
        encoder_args = {'n_filters': [128, 128, 256, 512],
                        'filter_sizes': [40, 20, 10, 10],
                        'strides': [1, 2, 2, 1]
                        }
    else:
        assert(False)

    n_input = [n_pc_points, 3]

    decoder_args = {'layer_sizes': [1024, 2048, np.prod(n_input)]}

    res = {'encoder': encoder_with_convs_and_symmetry,
           'decoder': decoder_with_fc_only,
           'encoder_args': encoder_args,
           'decoder_args': decoder_args
           }
    return res


def default_train_params_ala_nips_17(single_class=True):
    if single_class:
        params = {'batch_size': 50,
                  'training_epochs': 1000,
                  'denoising': False,
                  'learning_rate': 0.0005,
                  'z_rotate': False,
                  'saver_step': 10,
                  'loss_display_step': 1
                  }
    else:
        assert(False)
    return params
