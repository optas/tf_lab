import numpy as np
import os.path as osp
from shutil import copyfile

from general_tools.in_out import create_dir
from general_tools.plotting import stack_images_horizontally, stack_images_in_square_grid


def find_first_distant_enough(dists, dist_thres=0.01):
    for i, d in enumerate(dists):
        if d > dist_thres:
            return i
    return -1

def sorted_indices_given_exclusions(source, all_pdists, excluded):
    s_dists = all_pdists[source, :]
    s_dists[source] = np.inf
    s_dists[excluded] = np.inf
    return np.argsort(s_dists)

def far_from_two_observations(all_dists, obs_1, obs_2, excluded):
    n = all_pdists.shape[0]
    candidates = np.arange(n)
    candidates = np.setdiff1d(candidates, excluded)
    candidates = np.setdiff1d(candidates, [obs_1, obs_2])        
    sum_dist = all_pdists[obs_1, candidates] + all_pdists[obs_2, candidates]
    aso = np.argsort(sum_dist)
    aso = candidates[aso]
    return aso
    
def make_triplets(sources, all_pdists, excluded, rule, far_threshold=0.9):
    n = all_pdists.shape[0]
    triplets = []
    for i, s in enumerate(sources):
        candidates = np.arange(n)
        candidates = np.setdiff1d(candidates, excluded)
        candidates = np.setdiff1d(candidates, s)        
        aso = np.argsort(all_pdists[s, candidates])                        
        aso = candidates[aso]
       
        if rule == 'closest_nn':                             
            triplets.append([s, aso[0], aso[1]])
        
        elif rule == 'one_far':
            far_p = int(np.round(len(aso) * far_threshold))
            triplets.append([s, aso[0], aso[far_p]])
        
        elif rule == 'both_far':
            far_p = int(np.round(len(aso) * far_threshold))            
            aso_2 = far_from_two_observations(all_pdists, s, far_p, excluded)
            far_p2 = int(np.round(len(aso_2) * far_threshold))            
            triplets.append([s, aso_2[far_p2], aso[far_p]])            
        else:
            assert(False)                
    return triplets


def save_triplets(triplets, top_image_dir, top_out_dir, model_ids):
    image_view_tag = 'image_p020_t337_r005.png'
    logger = open(osp.join(top_out_dir, 'model_names_of_triplets.txt'), 'w')
    for i, t in enumerate(triplets):
        logger.write(str(i) + '\t')
        for k in range(3):
            save_dir = create_dir(osp.join(top_out_dir, str(i)))
            source = osp.join(top_image_dir, model_ids[t[k]], image_view_tag)
            dest = osp.join(save_dir, str(k) + '_' + model_ids[t[k]] + '.png')            
            copyfile(source, dest)
            if k < 2:
                logger.write(model_ids[t[k]] + '\t')
            else:
                logger.write(model_ids[t[k]] + '\n')
    
    logger.close()
    
def plot_triplets(triplets, top_image_dir, top_out_dir, model_ids):
    image_view_tag = 'image_p020_t337_r005.png'
    for i, t in enumerate(triplets):
        image_files = []
        image_files.append(osp.join(top_image_dir, model_ids[t[0]], image_view_tag))
        image_files.append(osp.join(top_image_dir, model_ids[t[1]], image_view_tag))
        image_files.append(osp.join(top_image_dir, model_ids[t[2]], image_view_tag))
        save_file = osp.join(top_out_dir, str(i) + '.png')
        stack_images_horizontally(image_files, save_file=save_file)
        
def plot_triplets_in_multiple_contexts(triplets, top_image_dir, top_out_dir, model_ids):
    image_view_tag = 'image_p020_t337_r005.png'
    n_contexts = len(triplets)
    n_examples = len(triplets[0])
    for i in range(n_examples):        
        image_files = []
        for context in range(n_contexts):                    
            t = triplets[context][i]
            image_files.append(osp.join(top_image_dir, model_ids[t[0]], image_view_tag))
            image_files.append(osp.join(top_image_dir, model_ids[t[1]], image_view_tag))
            image_files.append(osp.join(top_image_dir, model_ids[t[2]], image_view_tag))
        save_file = osp.join(top_out_dir, str(i) + '.png')
        stack_images_in_square_grid(image_files, save_file=save_file)