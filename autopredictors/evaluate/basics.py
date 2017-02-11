import glob
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np

from general_tools.in_out.basics import  create_dir
from geo_tool import Point_Cloud

from .. plotting.basics import  plot_original_pclouds_vs_reconstructed


def read_saved_epochs(saved_dir):
    epochs_saved = []
    files = glob.glob(osp.join(saved_dir, 'models.ckpt-*.index'))
    for f in files:
        epochs_saved.append(int(osp.basename(f)[len('models.ckpt-'):-len('.index')]))
        epochs_saved.sort()
    return epochs_saved


def generalization_error(model, train_data, test_data, val_data, configuration):
    conf = configuration
    epochs_saved = read_saved_epochs(conf.train_dir)
    stats = np.zeros((len(epochs_saved), 4))
    for iter, epoch in enumerate(epochs_saved):
        model.restore_model(conf.train_dir, epoch)
        _, l_tr, _ = model.evaluate(train_data, conf)
        _, l_te, _ = model.evaluate(test_data, conf)
        _, l_va, _ = model.evaluate(val_data, conf)
        stats[iter,:] = [epoch, l_tr, l_te, l_va]
        print(stats[iter,:])

        if iter == 0:
            gen_error = l_va - l_tr
            best_iter = iter
        elif (l_va - l_tr) < gen_error:
             gen_error = l_va - l_tr
             best_iter = iter

    gen_error = stats[best_iter, 2] - stats[best_iter, 1]
    best_epoch = int(stats[best_iter, 0])
    return gen_error, best_epoch, stats
