'''
Created on February 4, 2017

@author: optas
'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, conv_2d, avg_pool_1d, highway_conv_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected

from . utils import pairwise_distance, get_edge_feature, knn, soft_maxed_edge, pdist
#from . point_net_pp.modules import pointnet_pp_module
from .. fundamentals.layers import conv_1d_tranpose
from .. fundamentals.utils import expand_scope_by_name, replicate_parameter_for_all_layers

# For rotation transformers:
from with_others.million_geometries.src.rotations import octahedral_rotation_group, rotation_from_degrees
from with_others.million_geometries.src.hacks import bulb_pooling
    
    
def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=[False], spn=False, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d, container=None):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    b_norm = replicate_parameter_for_all_layers(b_norm, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    if spn:
        print 'Spatial transformer will be used.'
        raise ValueError()
        with tf.variable_scope('spn') as sub_scope:
            pool_size = 20  # how many points in each pool
            n_rotations = len(octahedral_rotation_group())
            R = pc_rotation_predictor(in_signal, pool_size, n_rotations, container=container)
            
            #in_signal = tf.matmul(in_signal, R)
            if container is not None:
                container['signal_transformed'] = tf.matmul(in_signal, R)
                
    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm[i]:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] != 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print layer

    if closing is not None:
        layer = closing(layer)
        print layer

    return layer


def encoder_with_convs_and_symmetry_new():
    # TODO delete after patching all AEs.
    pass
    
    
def cluster_pool(batch_size, in_features, n_clusters):
    ''' read again capsules.
    '''
    class_scores = conv_1d(in_features, nb_filter=n_clusters, filter_size=1)    
    class_scores = tf.nn.softmax(class_scores)    
    res = []
    for i in range(batch_size):  # TODO - implement nicely.
        _, nn_idx = tf.nn.top_k(class_scores[i], k=1)    # << This is not differentiable.
        nn_idx = tf.squeeze(nn_idx, -1)    
        class_aggregate = tf.unsorted_segment_max(in_features[i], nn_idx, n_clusters)
        res.append(class_aggregate)
    res = tf.stack(res)
    return res

    
def encoder_with_dynamic_edge_convolutions(in_signal, n_filters, filter_sizes=[1], strides=[1], neighbs=[20], b_norm=[True],
                                           regularizer=None, weight_decay=0.001, conv_op=conv_2d, symmetry=tf.reduce_max,
                                           non_linearity=tf.nn.relu,
                                           padding='same', reuse=False, scope=None, verbose=False):
    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')
    
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    neighb_sizes = replicate_parameter_for_all_layers(neighbs, n_layers)
    b_norms = replicate_parameter_for_all_layers(b_norm, n_layers)

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        # conv_op = conv_1d
        # TODO read again Implementation.
        adj_matrix = pairwise_distance(layer)
        nn_idx = knn(adj_matrix, k=neighb_sizes[i])
        layer = get_edge_feature(layer, nn_idx=nn_idx, k=neighb_sizes[i])
        
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)
        
        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norms[i]:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)
        

        
        if symmetry is not None:
            layer = symmetry(layer, axis=2)

        if verbose:
            print layer

        
        # batch_size = 50
        # if neighb_sizes[i] > 0:
        #     layer = cluster_pool(batch_size, layer, neighb_sizes[i])
        # print layer

    if symmetry is not None:
        layer = symmetry(layer, axis=1)    
    return layer



def encoder_with_grouping_and_interpolation(in_signal, grp_config=None, interp_config=None, b_norm=True, bn_decay=None, use_normal=False, scope=None, reuse=False, is_training=None):            

    sa_fp=pointnet_pp_module(in_signal, grp_config.filters, grp_config.points, grp_config.radii, grp_config.samples, interp_config.filters, interp_config.idx, interp_config.use_pts0, return_all=interp_config.return_all, b_norm=b_norm, bn_decay=bn_decay, use_normal=use_normal, scope=scope, reuse=reuse, is_training=is_training)
    #note, if no interp layers, we need to squeeze the pointwise features (1,n_dim) to (n_dim)
    sa_fp=tf.reshape(sa_fp,[-1,sa_fp.get_shape()[-1]])
    return sa_fp


def encoder_with_covns_and_grouping(reuse=False, scope=None):
    '''
        Point-Net++ encoder.
    '''
    pass


def decoder_with_fc_only(latent_signal, layer_sizes, b_norm=[True], non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False, container=None):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    
    if verbose:
        print 'Building Decoder'

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers-1)
    b_norm = replicate_parameter_for_all_layers(b_norm, n_layers-1)
    
    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
        
        if container is not None:
            container.append(layer)
            
        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm[i]:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if container is not None:
            container.append(layer)
            
        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'
        
        if container is not None:
            container.append(layer)

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'
            
     
    return layer


def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=[True], non_linearity=tf.nn.relu,
                            conv_op=conv_1d_tranpose, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):

    if verbose:
        print 'Building Decoder'
    
    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers-1)
    b_norm = replicate_parameter_for_all_layers(b_norm, n_layers-1)

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer


# SPATIAL TRANSFORMERS: they are here because they rely on encoder_decorer functions (circlurlar dependency)

def pc_rotation_transformer(in_signal, n_pools, scope=None):
    '''Does regression on degrees.
    '''
    k = 6    
    feat_dim = 16
    feat = intrinsic_knn_distances(in_signal, k)    
    feat = encoder_with_convs_and_symmetry(feat, [32, 64, 64, feat_dim], symmetry=None, scope=scope)
    
    if n_pools is not None:
        pool_index, n_bulbs = bulb_pools(in_signal, n_pools)
        bulb_feats = tf.gather_nd(feat, pool_index)
        print bulb_feats
        bulb_feats = tf.reshape(bulb_feats, [-1, n_bulbs, n_pools, feat_dim])        
        bulb_feats = tf.reduce_max(bulb_feats, axis=-2)
        print bulb_feats
    
    if n_pools is None:
        bulb_weights = bulb_pools(in_signal, n_pools)
        n_bulbs = int(bulb_weights.shape[1])
        yo = tf.tile(tf.expand_dims(bulb_weights, -1), [1, 1, 1, feat_dim])
        ya = tf.tile(tf.expand_dims(feat, 1), [1, n_bulbs, 1, 1])
        bulb_feats = tf.reduce_max(tf.multiply(yo, ya), axis=-2)
        print bulb_feats

    feats = decoder_with_fc_only(bulb_feats, layer_sizes=[32, 32, 3], scope=scope)
    R = rotation_explicit_transformer_degrees(feats)
    
    return R, feats, bulb_feats


def pc_rotation_predictor(in_pc, pool_size, n_rotations, container=None, scope=None):
    ''' does prediction of angles by guessing one of from a predifined set of them (e.g. octahedral).
    '''
    batch_size = tf.shape(in_pc)[0]
    emb_dim = 32
    
    pool_index, _, n_bulbs = bulb_pooling(in_pc, pool_size)
    b_pools = tf.gather_nd(in_pc, pool_index)
    bulb_pcs = tf.reshape(b_pools, [-1, n_bulbs, pool_size, 3])
    means = tf.reduce_mean(bulb_pcs, axis=-2, keep_dims=True)
    bulb_pcs -= means
    bulb_pcs = tf.reshape(bulb_pcs, [-1, n_bulbs * pool_size, 3])        
    bulb_feats = encoder_with_convs_and_symmetry(bulb_pcs, [64, 64, 128, emb_dim], 
                                                 symmetry=None, scope=scope)
    
    bulb_feats = tf.reshape(bulb_feats, [-1, n_bulbs, pool_size, emb_dim])
    bulb_feats = tf.reduce_max(bulb_feats, axis=2)
        
    aggregate_feats = decoder_with_fc_only(bulb_feats,
                                           layer_sizes=[32, 32, 32, n_rotations], 
                                           scope=scope)
    probs = tf.nn.softmax(aggregate_feats)
    
    sharp_loss = tf.reduce_mean(-tf.log(tf.reduce_max(probs, 1)))
    #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, sharp_loss)

    probs_tile = tf.tile(tf.expand_dims(probs, 2), [1, 1, 3])  # expand to multpily three angles.
    rotation_group = tf.constant(octahedral_rotation_group())
    rotation_group = tf.tile(tf.expand_dims(rotation_group, 0), [batch_size, 1, 1])
    
    pred = tf.reduce_sum(tf.multiply(probs_tile, rotation_group), axis=1) # MAX or mean?
    R = rotation_from_degrees(pred)
    
    if container is not None:
        container['predicted_angles'] = pred
        container['r_probs'] = probs
        container['rotation'] = R
        container['sharp_loss'] = sharp_loss        
    return R