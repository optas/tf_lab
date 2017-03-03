import glob
import os.path as osp
import numpy as np
from sklearn.neighbors import NearestNeighbors


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


def accuracy_of_completion(pred_pcloud, gt_pcloud, thres=0.02, ret_dists=False):
    '''Returns the fraction of points in the predicted point-cloud that are within
    `thres` euclidean distance from any point in the ground-truth point-cloud.
    '''
    nn = NearestNeighbors().fit(gt_pcloud)
    indices = nn.radius_neighbors(pred_pcloud, radius=thres, return_distance=False)
    success_indicator = [i.size >= 1 for i in indices]
    score = np.sum(success_indicator) / float(len(pred_pcloud))
    dists = None
    if ret_dists:
        dists, _ = nn.kneighbors(pred_pcloud, n_neighbors=1)

    return score, dists


def coverage_of_completion(gt_pcloud, pred_pcloud, thres=0.02, ret_dists=False):
    '''Returns the fraction of points in the ground-truth point-cloud that are within
    `thres` euclidean distance from any point in the predicted point-cloud.
    '''
    nn = NearestNeighbors().fit(pred_pcloud)
    indices = nn.radius_neighbors(gt_pcloud, radius=thres, return_distance=False)
    success_indicator = [i.size >= 1 for i in indices]
    score = np.sum(success_indicator) / float(len(gt_pcloud))
    dists = None
    if ret_dists:
        dists, _ = nn.kneighbors(gt_pcloud, n_neighbors=1)

    return score, dists


def l1_loss_comparison_like_Angela(gt_df, pred_df, unknown_space_mask, ignore_range=3.0):
    close_enough_mask = np.logical_or(pred_df < ignore_range, gt_df < ignore_range)
    total_mask = close_enough_mask * unknown_space_mask
    return np.sum(np.abs(gt_df - pred_df) * total_mask) / np.sum(total_mask)
