import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from general_tools.in_out.basics import create_dir
from geo_tool import Point_Cloud


def plot_original_pclouds_vs_reconstructed(feed_batches, recon_batches, gt_of_batches, save_dir, max_plot=None):
    '''Example:
    ae.restore_model(conf.train_dir, epoch)
    reconstructed, loss, original = ae.evaluate(train_data, conf)
    original_pclouds_vs_reconstructed(original, reconstructed, out_dir)
    '''
    def plot_feed_and_reconstructions():
        counter = 0
        for ob, rb in zip(feed_batches, recon_batches):     # Iterate over batches.
            for oi, ol, ri, rl in zip(ob[0], ob[1], rb[0], rb[1]):      # Iterate over pclouds inside batch (pcloud-label).
                if ol != rl:
                    raise ValueError()
                counter += 1
                if max_plot and counter > max_plot:
                    return

                fig = Point_Cloud(points=oi).plot(show=False)
                fig.savefig(osp.join(save_dir, '%s_feed.png' % (ol, )))
                plt.close()

                fig = Point_Cloud(points=ri).plot(show=False)
                fig.savefig(osp.join(save_dir, '%s_reconstructed.png' % (rl, )))
                plt.close()

    def plot_gt_of_feed():
        counter = 0
        if gt_of_batches is not None:
            for batch in gt_of_batches:
                for pc, l in zip(batch[0], batch[1]):
                    counter += 1
                    if max_plot and counter > max_plot:
                        return
                    fig = Point_Cloud(points=pc).plot(show=False)
                    fig.savefig(osp.join(save_dir, '%s_gt_feed.png' % (l, )))
                    plt.close()

    plt.ioff()
    create_dir(save_dir)
    plot_feed_and_reconstructions()
    plot_gt_of_feed()


def plot_train_val_test_curves(stats, save_dir, has_validation=True, best_epoch=None, show=True):
    create_dir(save_dir)
    n_epochs = stats.shape[0]
    x = range(n_epochs)
    fig, ax = plt.subplots()
    plt.plot(x, stats[:, 1])
    plt.plot(x, stats[:, 2])
    if has_validation:
        plt.plot(x, stats[:, 3])
    plt.xlabel('Epochs')
    plt.ylabel('AE Loss')
    if has_validation:
        plt.legend(['Train', 'Test', 'Val'])
    else:
        plt.legend(['Train', 'Test'])

    xticks = x[0: len(x): n_epochs / 10]
    ax.set_xticks(xticks)
    ax.set_xticklabels(stats[:, 0].astype(np.int16)[xticks])    # TODO - Cleaner tick-set

    if best_epoch is not None:
        fig.suptitle('Best-epoch w.r.t. GE = ' + str(best_epoch))

    if show:
        plt.show()
    if has_validation:
        tag = 'train-test-val-curves.png'
    else:
        tag = 'train-test-curves.png'
    fig.savefig(osp.join(save_dir, tag))


def plot_reconstructions_at_epoch(epoch, model, in_data, configuration, save_dir, max_plot=None):
    conf = configuration
    model.restore_model(conf.train_dir, epoch)
    reconstructed, _, feed, gt_feed = model.evaluate(in_data, conf)
    plot_original_pclouds_vs_reconstructed(feed, reconstructed, gt_feed, save_dir, max_plot=max_plot)
