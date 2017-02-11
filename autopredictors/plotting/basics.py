import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from general_tools.in_out.basics import  create_dir
from geo_tool import Point_Cloud


def plot_original_pclouds_vs_reconstructed(original_batches, recon_batches, save_dir, max_plot=None):
    '''Example: 
        ae.restore_model(conf.train_dir, epoch)
        reconstructed, loss, original = ae.evaluate(train_data, conf)
        original_pclouds_vs_reconstructed(original, reconstructed, out_dir)
    '''
    counter = 0
    plt.ioff()
    create_dir(save_dir)
    for ob, rb in zip(original_batches, recon_batches):
        for oi, ri in zip(ob, rb):
            counter += 1            
            if max_plot and counter > max_plot:
                return
            
            fig = Point_Cloud(points=oi[0]).plot(show=False);
            fig.savefig(osp.join(save_dir, '%s_original.png' % (oi[1], ) ))
            plt.close()
            
            fig = Point_Cloud(points=ri[0]).plot(show=False);    
            fig.savefig(osp.join(save_dir, '%s_reconstructed.png' % (ri[1], ) ))
            plt.close()


def plot_train_val_test_curves(stats, save_dir, best_epoch=None, show=True):
    create_dir(save_dir)
    n_epochs = stats.shape[0]
    x = range(n_epochs)
    fig, ax = plt.subplots()
    plt.plot(x, stats[:, 1])
    plt.plot(x, stats[:, 2])
    plt.plot(x, stats[:, 3])
    plt.xlabel('Epochs')
    plt.ylabel('AE Loss')
    plt.legend(['Train', 'Test', 'Val'])    
    
    xticks = x[0 : len(x) : n_epochs/10]
    ax.set_xticks(xticks)
    ax.set_xticklabels(stats[:, 0].astype(np.int16)[xticks]) # TODO - Cleaner tick-set
    
    if best_epoch is not None:
        fig.suptitle('Best-epoch w.r.t. GE = ' + str(best_epoch))
    
    if show:
        plt.show()
    fig.savefig(osp.join(save_dir, 'train-test-val-curves.png'))
    

def plot_reconstructions_at_epoch(epoch, model, in_data, configuration, save_dir, max_plot=None):
    conf = configuration
    model.restore_model(conf.train_dir, epoch)
    reconstructed, loss, original = model.evaluate(in_data, conf)
    plot_original_pclouds_vs_reconstructed(original, reconstructed, save_dir, max_plot=max_plot)    