'''
Created on Sep 7, 2017

@author: optas
'''


def permissible_dictionary(file_with_ids):
    ''' Returns a dictionary with model_ids that are white-listed in the input file.
    '''
    data_dict = defaultdict(dict)
    with open(file_with_ids, 'r') as f_in:
        for line in f_in:
            syn_id, model_id, scan_id = line.split()
            if len(data_dict[syn_id]) == 0:
                data_dict[syn_id] = set()
            data_dict[syn_id].add((model_id + scan_id))
    return data_dict


def mask_of_permissible(model_names, permissible_file, class_syn_id=None):
    ''' model_names: N x 1 np.array
        returns : mask N x 1 boolean'''
    permissible_dict = permissible_dictionary(permissible_file)
    if class_syn_id is not None:
        permissible_dict = permissible_dict[class_syn_id]

    mask = np.zeros([len(model_names)], dtype=np.bool)
    for i, model_id in enumerate(model_names):
        if model_id in permissible_dict:
            mask[i] = True

    return mask