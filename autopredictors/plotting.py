import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from general_tools.in_out.basics import create_dir
from geo_tool import Point_Cloud


def plot_original_pclouds_vs_reconstructed(reconstructions, feed_data, ids, original_data, save_dir, data_loss=None, in_u_sphere=False, max_plot=None):
    '''Example:
    ae.restore_model(conf.train_dir, epoch)
    reconstructions, feed_data, ids, original_data = ae.evaluate(train_data, conf)
    original_pclouds_vs_reconstructed(original, reconstructed, out_dir)
    '''
    create_dir(save_dir)
    plt.ioff()
    counter = 0
    for recon, feed, original, idx in zip(reconstructions, feed_data, original_data, ids):
        if max_plot and counter > max_plot:
            return

        fig = Point_Cloud(points=recon).plot(show=False, in_u_sphere=in_u_sphere)
        if data_loss is not None:
            fig.axes[0].title.set_text('Prediction with loss = %f.' % data_loss[counter])
        fig.savefig(osp.join(save_dir, '%s_reconstructed.png' % (idx, )))
        plt.close()
        fig = Point_Cloud(points=feed).plot(show=False, in_u_sphere=in_u_sphere)
        fig.savefig(osp.join(save_dir, '%s_feed.png' % (idx, )))
        plt.close()
        fig = Point_Cloud(points=original).plot(show=False, in_u_sphere=in_u_sphere)
        fig.savefig(osp.join(save_dir, '%s_feed_gt.png' % (idx, )))
        plt.close()
        counter += 1


def plot_train_val_test_curves(stats, save_dir=None, has_validation=True, best_epoch=None, show=True):
    create_dir(save_dir)
    n_epochs = stats.shape[0]
    x = range(n_epochs)
    fig, ax = plt.subplots()
    plt.plot(x, stats[:, 1])
    plt.plot(x, stats[:, 2])

    if has_validation:
        plt.plot(x, stats[:, 3])
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')

    if has_validation:
        plt.legend(['Train', 'Val', 'Test'])
    else:
        plt.legend(['Train', 'Test'])

    tick_step = max(1, n_epochs / 10)
    xticks = x[0: len(x): tick_step]
    ax.set_xticks(xticks)
    ax.set_xticklabels(stats[:, 0].astype(np.int16)[xticks])    # TODO - Cleaner tick-set

    if best_epoch is not None:
        fig.suptitle('Best-epoch w.r.t. GE = ' + str(best_epoch))

    if show:
        plt.show()
    if has_validation:
        tag = 'train-val-test-curves.png'
    else:
        tag = 'train-test-curves.png'

    if save_dir is not None:
        fig.savefig(osp.join(save_dir, tag))


def plot_reconstructions_at_epoch(epoch, model, in_data, configuration, save_dir, in_u_sphere=False, max_plot=None):
    conf = configuration
    model.restore_model(conf.train_dir, epoch)
    reconstructions, losses, feed_data, ids, original_data = model.evaluate_one_by_one(in_data, conf)
    plot_original_pclouds_vs_reconstructed(reconstructions, feed_data, ids, original_data, save_dir, data_loss=losses, in_u_sphere=in_u_sphere, max_plot=max_plot)
