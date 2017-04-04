'''
Created on Mar 32, 2017

@author: optas

Associate code and data for manipulating the shaped of Model-Net-10 (and 40).

'''

from geo_tool.in_out.soup import load_ply


def pc_loader(f_name):
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id