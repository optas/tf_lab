import glob
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


def eval_model(model, configuration, datasets, epochs=None, verbose=False):
    conf = configuration
    if type(datasets) != list:
        datasets = [datasets]

    if epochs is None:
        epochs = read_saved_epochs(conf.train_dir)

    stats = np.zeros((len(epochs), len(datasets)))
    for i, epoch in enumerate(epochs):
        model.restore_model(conf.train_dir, epoch, verbose)
        for j, d in enumerate(datasets):
            loss = model.evaluate(d, conf)[1]
            stats[i, j] = loss
        if verbose:
            print stats[i, :]

    epochs = np.array(epochs).reshape((len(epochs), 1))
    stats = np.hstack((epochs, stats))
    return stats


def generalization_error(model, train_data, test_data, val_data, configuration):
    conf = configuration
    epochs_saved = read_saved_epochs(conf.train_dir)
    stats = np.zeros((len(epochs_saved), 4))

    for i, epoch in enumerate(epochs_saved):
        model.restore_model(conf.train_dir, epoch)
        l_tr = model.evaluate(train_data, conf)[1]
        l_te = model.evaluate(test_data, conf)[1]
        l_va = model.evaluate(val_data, conf)[1]
        stats[i, :] = [epoch, l_tr, l_te, l_va]
        print(stats[i, :])

        if i == 0:
            gen_error = l_va - l_tr
            best_iter = i
        elif (l_va - l_tr) < gen_error:
            gen_error = l_va - l_tr
            best_iter = i

    gen_error = stats[best_iter, 2] - stats[best_iter, 1]
    best_epoch = int(stats[best_iter, 0])
    return gen_error, best_epoch, stats
