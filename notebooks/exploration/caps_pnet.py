def adaptive_hierarchical_encoder_1(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                  b_norm=True, spn=False, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                  symmetry=tf.reduce_max, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                  reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d):

    if verbose:
        print 'Building Encoder'

    n_pc_points = int(in_signal.shape[1])
    batch_size = tf.shape(in_signal)[0]
    
#     with tf.name_scope("coupling_weights"):
#         W_couple = tf.random_normal(shape=(1, n_pc_points, n_pc_points), stddev=0.0001, dtype=tf.float32, name="W_coupling") #         W = tf.Variable(W_couple, name="W_couple")
#         coupler = tf.nn.softmax(W, dim=2, name="soft-max-coupling")
#         coupler = tf.tile(coupler, [batch_size, 1, 1])
    
    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')
    
    codes = []
    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal
        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i],
                        regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)
        
        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list())+np.prod(layer.gamma.get_shape().as_list())

        
            # NEW
            layer = layer - tf.reduce_mean(layer, axis=[1], keep_dims=True)
            cov_layer_i = tf.matmul(layer, layer, transpose_b=True)
            cov_layer_i = tf.nn.softmax(cov_layer_i, dim=1, name="soft-max-coupling")
            cov_layer_i = tf.matrix_set_diag(cov_layer_i, tf.zeros(shape=(batch_size, n_pc_points)))                    
            layer = tf.matmul(cov_layer_i, layer)
            
            
        if non_linearity is not None:
            layer = non_linearity(layer)

#         layer = tf.matmul(coupler, layer)
        codes.append(tf.reduce_max(layer, axis=1))
        
        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

#     if symmetry is not None:
#         layer = symmetry(layer, axis=1)
#         if verbose:
#             print layer

    return tf.concat(codes, axis=1)