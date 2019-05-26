"""
High level functions that compute distances among features/vector spaces. 

Created on May 25, 2019

@author: optas
"""


from general_tools import iterate_in_chunks

def compute_k_neighbors(A_feat, B_feat, sess, k=200, batch_size=512, sim='cosine'):
    n_a, n_dims = A_feat.shape
    n_b, dummy = B_feat.shape
    assert(n_dims == dummy)
    
    if sim == 'cosine':        
        sims, index, A_pl, B_pl = cosine_k_neighbors(n_dims, k)
    elif sim == 'euclid':
        sims, index, A_pl, B_pl = euclidean_k_neighbors(n_dims, k)
    else:
        raise ValueError('Uknown distance.')
    
    s = []  # similarities
    i = []  # neighbors
    
    for idx in iterate_in_chunks(np.arange(n_a), batch_size):        
        r = sess.run([sims, index], feed_dict={A_pl:A_feat[idx], B_pl:B_feat})
        s.append(r[0])
        i.append(r[1])
    return np.vstack(s), np.vstack(i)