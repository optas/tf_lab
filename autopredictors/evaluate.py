import glob
import os.path as osp
import numpy as np
import warnings
from scipy.stats import hmean
from . scripts.helper import shape_net_core_synth_id_to_category

try:
    from sklearn.neighbors import NearestNeighbors
except:
    warnings.warn('Sklearn library is not installed.')

from geo_tool import Point_Cloud
from general_tools.in_out.basics import create_dir
from collections import defaultdict


def read_saved_epochs(saved_dir):
    epochs_saved = []
    files = glob.glob(osp.join(saved_dir, 'models.ckpt-*.index'))
    for f in files:
        epochs_saved.append(int(osp.basename(f)[len('models.ckpt-'):-len('.index')]))
        epochs_saved.sort()
    return epochs_saved


def save_reconstructions(out_dir, reconstructions, gt_data, feed_data, ids):
    create_dir(out_dir)
    for rpc, gpc, fpc, name in zip(reconstructions, gt_data, feed_data, ids):
        save_id = osp.join(out_dir, name)
        Point_Cloud(points=rpc).save_as_ply(save_id + '_prediction')
        Point_Cloud(points=gpc).save_as_ply(save_id + '_gt')
        Point_Cloud(points=fpc).save_as_ply(save_id + '_feed')


def save_pc_prediction_stats(file_out, data_ids, scores):
    accurasy = scores[:, 0]
    coverage = scores[:, 1]
    hmeas = hmean(scores, axis=1)
    splitted_ids = [' '.join(t.split('.')) for t in data_ids]
    with open(file_out, 'w') as fout:
        fout.write('Mode ID: Accuracy Coverage H-Measure\n')
        for i, sid in enumerate(splitted_ids):
            fout.write(sid + '%f %f %f\n' % (accurasy[i], coverage[i], hmeas[i]))


def save_stats_of_multi_class_experiments(file_out, test_ids, pred_scores):
    splitted_ids = [t.split('.') for t in test_ids]
    pes_class_accurasy = defaultdict(list)
    pes_class_coverage = defaultdict(list)
    pes_class_hmean = defaultdict(list)
    hmeas = hmean(pred_scores, axis=1)

    for i, s in enumerate(splitted_ids):
        class_id = s[0]
        pes_class_accurasy[class_id].append(pred_scores[i, 0])
        pes_class_coverage[class_id].append(pred_scores[i, 1])
        pes_class_hmean[class_id].append(hmeas[i])

    with open(file_out, 'w') as fout:
        fout.write('Class Accuracy Coverage H-Measure\n')
        for class_id in pes_class_accurasy.keys():
            acc = np.median(pes_class_accurasy[class_id])
            cov = np.median(pes_class_coverage[class_id])
            hm = np.median(pes_class_hmean[class_id])
            fout.write(shape_net_core_synth_id_to_category[class_id] + ' %f %f %f\n' % (acc, cov, hm))


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


def accuracy_of_completion(pred_pcloud, gt_pcloud, thres=0.02, ret_dists=False):
    '''Returns the fraction of points in the predicted point-cloud that are within
    `thres` euclidean distance from any point in the ground-truth point-cloud.
    '''
    nn = NearestNeighbors().fit(gt_pcloud)
    indices = nn.radius_neighbors(pred_pcloud, radius=thres, return_distance=False)
    success_indicator = [i.size >= 1 for i in indices]
    score = np.sum(success_indicator) / float(len(pred_pcloud))

    if ret_dists:
        dists, _ = nn.kneighbors(pred_pcloud, n_neighbors=1)
        return score, dists
    else:
        return score


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
    else:
        return score


def l1_loss_comparison_like_Angela(gt_df, pred_df, unknown_space_mask, ignore_range=None):
    if ignore_range is not None:
        close_enough_mask = gt_df < ignore_range
    else:
        close_enough_mask = np.ones(pred_df.shape, dtype=np.bool)

    total_mask = close_enough_mask * unknown_space_mask
    return np.sum(np.abs(gt_df - pred_df) * total_mask) / np.sum(total_mask)


def paper_pc_completion_experiment_id_best_epoch(category, loss):
    experiment_id = {"airplane": 1, "chair": 2, "car": 3, "table": 4, "vessel": 5, "lamp": 6, "sofa": 7, "cabinet": 8}
    best_epoch = dict()
    best_epoch['chamfer'] = {"airplane": 48, "chair": 40, "car": 26, "table": 46, "vessel": 40, "lamp": 30, "sofa": 42, "cabinet": 90, "all_classes": 73}
    best_epoch['emd'] = {"airplane": 54, "chair": 52, "car": 20, "table": 82, "vessel": 60, "lamp": 74, "sofa": 74, "cabinet": 76, "all_classes": 82}
    if category == "all_classes":
        if loss == "chamfer":
            res_exp_id = 10
        else:
            res_exp_id = 9
    else:
        res_exp_id = experiment_id[category]
    return res_exp_id, best_epoch[loss][category]
