def minimum_mathing_distance_old(sample_pcs, ref_pcs, batch_size, normalize=False, sess=None, verbose=False, use_sqrt=False, use_EMD=False):
    ''' normalize (boolean): if True the Chamfer distance between two point-clouds is the average of matched
                             point-distances. Alternatively, is their sum.
    '''
    if normalize:
        reducer = tf.reduce_mean
    else:
        reducer = tf.reduce_sum

    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible Point-Clouds.')

    # TF Graph Operations
    ref_pl = tf.placeholder(tf.float32, shape=(1, n_pc_points, pc_dim))
    sample_pl = tf.placeholder(tf.float32, shape=(None, n_pc_points, pc_dim))

    repeat_times = tf.shape(sample_pl)[0]   # slower- could be used to use entire set of samples.
#     repeat_times = batch_size
    ref_repeat = tf.tile(ref_pl, [repeat_times, 1, 1])
    ref_repeat = tf.reshape(ref_repeat, [repeat_times, n_pc_points, pc_dim])

    if not use_EMD:
        ref_to_s, _, s_to_ref, _ = nn_distance(ref_repeat, sample_pl)

        if use_sqrt:
            ref_to_s = tf.sqrt(ref_to_s)
            s_to_ref = tf.sqrt(s_to_ref)

        all_dist_in_batch = reducer(ref_to_s, 1) + reducer(s_to_ref, 1)
    else:
        match = approx_match(ref_repeat, sample_pl)
        all_dist_in_batch = match_cost(ref_repeat, sample_pl, match)

    best_in_batch = tf.reduce_min(all_dist_in_batch)   # Best distance, of those that were matched to single ref pc.
    matched_dists = []
    for i in xrange(n_ref):
        best_in_all_batches = []
        if verbose and i % 50 == 0:
            print i
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
#             if len(sample_chunk) != batch_size:
#                 continue
            feed_dict = {ref_pl: np.expand_dims(ref_pcs[i], 0), sample_pl: sample_chunk}
            b = sess.run(best_in_batch, feed_dict=feed_dict)
            best_in_all_batches.append(b)

        matched_dists.append(np.min(best_in_all_batches))

    mmd = np.mean(matched_dists)
    return mmd, matched_dists




def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                    b_norm=True, spn=False, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                    symmetry=tf.reduce_max, dropout_prob=None, scope=None, reuse=False):

    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''
    
    # THE guys from ICML-Workshop (AE-64bneck used in language use this one)
    warnings.warn('Using old architecture.')
    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    if spn:
        transformer = pcloud_spn(in_signal)
        in_signal = tf.batch_matmul(in_signal, transformer)
        print 'Spatial transformer was activated.'

    name = 'encoder_conv_layer_0'
    scope_i = expand_scope_by_name(scope, name)
    layer = conv_1d(in_signal, nb_filter=n_filters[0], filter_size=filter_sizes[0], strides=strides[0], regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i)

    if b_norm:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

    layer = non_linearity(layer)

    if dropout_prob is not None and dropout_prob[0] > 0:
        layer = dropout(layer, 1.0 - dropout_prob[0])

    for i in xrange(1, n_layers):
        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_1d(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer, weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i)

        if b_norm:
            name += '_bnorm'
            #scope_i = expand_scope_by_name(scope, name) # FORGOT TO PUT IT BEFORE ICLR
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)

        layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

    if symmetry is not None:
        layer = symmetry(layer, axis=1)

    return layer