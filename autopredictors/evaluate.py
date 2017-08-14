import glob
import os.path as osp
import numpy as np
import warnings
from scipy.stats import hmean
from collections import defaultdict

from geo_tool import Point_Cloud
from general_tools.in_out.basics import create_dir


try:
    from sklearn.neighbors import NearestNeighbors
except:
    warnings.warn('Sklearn library is not installed.')


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
