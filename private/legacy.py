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