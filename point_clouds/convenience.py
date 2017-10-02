'''
Created on September 5, 2017

@author: optas
'''
import numpy as np
from general_tools.simpletons import iterate_in_chunks


def reconstruct_pclouds(autoencoder, pclouds_feed, batch_size, pclouds_gt=None, compute_loss=True):
    recon_data = []
    loss = 0.
    last_examples_loss = 0.     # keep track of loss on last batch which potentially is smaller than batch_size
    n_last = 0.

    n_pclouds = len(pclouds_feed)
    if pclouds_gt is not None:
        if len(pclouds_gt) != n_pclouds:
            raise ValueError()

    n_batches = 0.0
    idx = np.arange(n_pclouds)

    for b in iterate_in_chunks(idx, batch_size):
        feed = pclouds_feed[b]
        if pclouds_gt is not None:
            gt = pclouds_gt[b]
        else:
            gt = None
        rec, loss_batch = autoencoder.reconstruct(feed, GT=gt, compute_loss=compute_loss)
        recon_data.append(rec)
        if compute_loss:
            if len(b) == batch_size:
                loss += loss_batch
            else:  # last index was smaller than batch_size
                last_examples_loss = loss_batch
                n_last = len(b)
        n_batches += 1

    if n_last == 0:
        loss /= n_batches
    else:
        loss = (loss * batch_size) + (last_examples_loss * n_last)
        loss /= ((n_batches - 1) * batch_size + n_last)

    return np.vstack(recon_data), loss


def get_latent_codes(autoencoder, pclouds, batch_size=100):
    latent_codes = []
    idx = np.arange(len(pclouds))
    for b in iterate_in_chunks(idx, batch_size):
        latent_codes.append(autoencoder.transform(pclouds[b]))
    return np.vstack(latent_codes)
