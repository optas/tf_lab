import numpy as np
import os
from multiprocessing import Pool

import struct
import scipy.io as sio


def fsdf_bin_parser(file_name,res = 32):
    with open(file_name,'rb')  as fin:
	output_grid = np.ndarray((res,res,res),np.float32)
	
	n_output = res * res * res
	output_grid_values = struct.unpack('f' * n_output, fin.read(4 * n_output))
	
	k = 0

	for d in xrange(res):
	    for w in xrange(res):
		for h in xrange(res):
		    output_grid[w,h,d] = output_grid_values[k]
		    k += 1

	return output_grid


def fsdf_bin_loader_n(filename,n_threads):
    point_cloud_list = []
    filename_list = []
    for model_path in open(filename):
        filename_list.append(model_path.strip())

    pclouds = np.empty([len(filename_list),32,32,32,1],dtype=np.float32)
    model_names = np.empty([len(filename_list)],dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(fsdf_bin_parser,filename_list)):
        pclouds[i,:,:,:,0],model_names[i] = data

    pool.close()
    pool.join()
    return pclouds, model_names