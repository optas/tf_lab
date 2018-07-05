'''
Created on Nov 2, 2017

@author: optas
'''

import tensorflow as tf

try:
    losses_found = True
    if tf.__version__ == '1.6.0':
	print '1.6'
    	from .. external.Chamfer_EMD_tf16.tf_nndistance import nn_distance
    	from .. external.Chamfer_EMD_tf16.tf_approxmatch import approx_match, match_cost
    else:
        from .. external.Chamfer_EMD_tf1plus.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_tf1plus.tf_approxmatch import approx_match, match_cost
except:
    losses_found = False
    print('External Losses (Chamfer-EMD) cannot be loaded.')


def losses():
    if losses_found:
        return nn_distance, approx_match, match_cost
    else:
        return None, None, None
