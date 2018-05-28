import tensorflow as tf
from tflearn.layers.conv import conv_2d

from tf_lab import Neural_Net
from tf_lab.fundamentals.utils import expand_scope_by_name
from tf_lab.point_clouds.encoders_decoders import decoder_with_fc_only


def diff_reconstructor_on_conf_and_area(n_cons, n_pc_points, tie_weights=True):
    'New for SIGGASIA'
    'Optimized for n_cons = 40'
    with tf.variable_scope('conv_based_reconstructor') as scope:
        area_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons))
        conf_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons))        
        labels_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points, 3))
        
        conv_siamese = []
        reuse = False
        for name, pl in zip(['area', 'conformal'], [area_pl, conf_pl]):
            if name == 'conformal' and tie_weights:
                reuse = True
            else:
                scope = expand_scope_by_name(scope, 'conformal')
                
            
            layer = tf.expand_dims(pl, -1)
            s = expand_scope_by_name(scope, 'conv1')
            layer = conv_2d(layer, nb_filter=20, filter_size=3, strides=2, 
                            activation='relu', scope=s, reuse=reuse)
            
            s = expand_scope_by_name(scope, 'conv2')
            layer = conv_2d(layer, nb_filter=20, filter_size=6, strides=2, 
                            activation='relu', scope=s, reuse=reuse)
            
            conv_siamese.append(layer)
             
        layer = tf.stack(conv_siamese, axis=1)
        net_out = decoder_with_fc_only(layer, layer_sizes=[64, 128, n_pc_points * 3], b_norm=False)
        #net_out = decoder_with_fc_only(layer, layer_sizes=[64, 128, n_pc_points * 3], b_norm=False)
        net_out = tf.reshape(net_out, [-1, n_pc_points, 3])
    return net_out, area_pl, conf_pl, labels_pl


def diff_reconstructor(n_cons, n_pc_points):
    'Optimized for n_cons = 40, on area-based maps'
    with tf.variable_scope('conv_based_reconstructor'):
        feed_pl = tf.placeholder(tf.float32, shape=(None, n_cons, n_cons))
        labels_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points, 3))
        layer = tf.expand_dims(feed_pl, -1)
        layer = conv_2d(layer, nb_filter=20, filter_size=3, strides=2, activation='relu')
        layer = conv_2d(layer, nb_filter=20, filter_size=6, strides=2, activation='relu')
        net_out = decoder_with_fc_only(layer, layer_sizes=[128, 128, n_pc_points * 3], b_norm=False)
        net_out = tf.reshape(net_out, [-1, n_pc_points, 3])
    return net_out, feed_pl, labels_pl
