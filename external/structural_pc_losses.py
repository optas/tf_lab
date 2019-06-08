'''
Created on Nov 2, 2017

@author: optas
'''

import tensorflow as tf

try:
    losses_found = True
    if tf.__version__ == '1.6.0':
        from .. external.Chamfer_EMD_tf16.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_tf16.tf_approxmatch import approx_match, match_cost
    elif tf.__version__ == '1.3.0':
        from .. external.Chamfer_EMD_tf1plus.tf_nndistance import nn_distance
        from .. external.Chamfer_EMD_tf1plus.tf_approxmatch import approx_match, match_cost
    elif tf.__version__ =='1.13.1':
        from .. external.Chamfer_EMD_tf1_13.nn_distance.tf_nndistance import nn_distance
        approx_match = None
        match_cost = None
    else:
        losses_found = False
        print('External Losses (Chamfer-EMD) cannot be loaded.')
except:
    losses_found = False
    print('External Losses (Chamfer-EMD) cannot be loaded.')


def losses():
    if losses_found:
        return nn_distance, approx_match, match_cost
    else:
        return None, None, None
