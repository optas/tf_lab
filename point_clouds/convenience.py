'''
Created on September 5, 2017

@author: optas
'''
import numpy as np
from general_tools.simpletons import iterate_in_chunks


def reconstruct_pclouds(autoencoder, pclouds, batch_size, compute_loss=True):
    recon_data = []
    loss = 0.
    n_input = list(pclouds[0].shape)
    n_batches = 0.0
    for b in iterate_in_chunks(pclouds, batch_size):
        feed = b.reshape([len(b)] + n_input)
        rec, loss_batch = autoencoder.reconstruct(feed, compute_loss=compute_loss)
        recon_data.append(rec)
        if compute_loss:
            loss += loss_batch
        n_batches += 1

    return np.vstack(recon_data), loss / n_batches
