import os.path as osp

top_sample_dir = '/orions4-zfs/projects/optas/DATA/OUT/iclr/synthetic_samples/'
top_evaluation_dir = '/orions4-zfs/projects/optas/DATA/OUT/iclr/evaluations/synthetic_data/'

def stored_synthetic_samples(class_name):
    sample_dir = {'l_gan_emd': osp.join(top_sample_dir, 'l_gan/l_gan_' + class_name + '_mlp_with_split_1pc_usampled_bnorm_on_encoder_only_emd_bneck_128'),
                  'l_gan_chamfer': osp.join(top_sample_dir, 'l_gan/l_gan_' + class_name + '_mlp_with_split_1pc_usampled_bnorm_on_encoder_only_chamfer_bneck_128'),
                  'l_w_gan_small': osp.join(top_sample_dir, 'l_w_gan/l_w_gan_'+ class_name + '_mlp_with_split_1pc_usampled_bnorm_on_encoder_only_emd_bneck_128_lgan_arch'),
                  'l_w_gan_large': osp.join(top_sample_dir, 'l_w_gan/l_w_gan_'+ class_name + '_mlp_with_split_1pc_usampled_bnorm_on_encoder_only_emd_bneck_128_lgan_arch_double_neurons'),
                  'r_gan': osp.join(top_sample_dir, 'r_gan/r_gan_' + class_name + '_mlp_disc_4_fc_gen_raw_gan_2048_pts'),
                  'gmm': osp.join(top_sample_dir, 'gmm/gmm_emd_' + class_name)
                 }
    
    return sample_dir


def find_best_model_in_metric_file(in_file, metric, sort_by='test'):    
    all_lines = []
    res = dict()
    with open(in_file, 'r') as fin:
        for line in fin:
            l = line.rstrip()
            if len(l) > 0:
                all_lines.append(l)
    
    current_model = None
    
    for line in all_lines:
        token = line.split()
        if token[0] not in ['train', 'test', 'val']:
            current_model = token[0]
        else:
            if 'mmd' in metric.lower():                
                split, metric_value, _ = token # split - mean - std
            else:
                split, metric_value = token
                
            metric_value = float(metric_value)
            res[(current_model, split)] = metric_value
    
    best_model = sorted([(r[0], res[r]) for r in res.keys() if r[1] == sort_by], key=lambda x:x[1])[0][0]    
    
    for s in ['train', 'test', 'val']:
        print s, best_model, res[(best_model, s)]
    return res
    