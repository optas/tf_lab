'''
Created on Aug 12, 2017

@author: optas
'''
import numpy as np
from .. in_out.basics import read_saved_epochs


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
            print(stats[i, :])

    epochs = np.array(epochs).reshape((len(epochs), 1))
    stats = np.hstack((epochs, stats))
    return stats


def generalization_error(model, train_data, val_data, test_data, configuration, epochs=None):
    conf = configuration

    if epochs is None:
        epochs = read_saved_epochs(conf.train_dir)

    stats = np.zeros((len(epochs), 4))

    for i, epoch in enumerate(epochs):
        model.restore_model(conf.train_dir, epoch)
        l_tr = model.evaluate(train_data, conf)[1]
        l_va = model.evaluate(val_data, conf)[1]
        l_te = model.evaluate(test_data, conf)[1]
        stats[i, :] = [epoch, l_tr, l_va, l_te]
        print(stats[i, :])

        if i == 0:
            gen_error = abs(l_va - l_tr)
            best_iter = i
        elif abs(l_va - l_tr) < gen_error:
            gen_error = abs(l_va - l_tr)
            best_iter = i

    gen_error = abs(stats[best_iter, 2] - stats[best_iter, 1])
    best_epoch = int(stats[best_iter, 0])
    return gen_error, best_epoch, stats
