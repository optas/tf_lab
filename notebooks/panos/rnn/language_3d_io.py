import numpy as np
import csv
import pandas
import pandas as pd
import os.path as osp
from pandas import DataFrame

class GeoWordDatum():
    def __init__(self, words, geometry_embedding, label, triplet_id):
        self.words = np.array(words)        
        self.geometry = geometry_embedding.copy()
        self.label = label
        self.triplet_id = triplet_id
        self.context_condition = self.triplet_id.split('_')[0]
        self.context_id = self.triplet_id.split('_')[1]

    def n_steps(self):
        return len(self.words)
    
#    def padded_embedded_words(self, max_seq_len):
#        pad_many = max_seq_len - self.n_steps()
#        if pad_many >= 0:
#            return np.pad(self.words, ((0, pad_many), (0,0)) , 'constant')
#        elif pad_many < 0:
#            return self.words[:max_seq_len,:]
        
    def geo_embedding(self):
        return self.geometry
        
    def mask(self, max_seq_len):
        mask = np.zeros(max_seq_len)
        mask[:self.n_steps()] = 1.0
        return mask
    
    def padded_words(self, max_seq_len):
        pad_many = max_seq_len - self.n_steps()
        if pad_many >= 0:
            return np.pad(self.words, (0, pad_many), 'constant')
        else:
            return self.words[:max_seq_len]
        
    
class GeoWordData():
    @staticmethod
    def count_geo_word_data(triplet_to_data_dic):
        c = 0
        for tr_id in triplet_to_data_dic:
            c += len(triplet_to_data_dic[tr_id])
        return c
    
    @staticmethod
    def organize_into_numpy_arrays(triplet_to_data_dic, max_seq_len=30):
        n_rows = GeoWordData.count_geo_word_data(triplet_to_data_dic)
        
        geo_feat_dim = 128        
        word_matrix = np.zeros(shape=(n_rows, max_seq_len))
        geo_matrix = np.zeros(shape=(n_rows, 3 * geo_feat_dim))
        label_matrix = np.zeros((n_rows, 3), dtype=np.int32)
        mask_matrix = np.zeros(shape=(n_rows, max_seq_len))
        condition_matrix = np.zeros(shape=(n_rows), dtype=object)

        i = 0
        for triplet_id in triplet_to_data_dic:
            for gw_datum in triplet_to_data_dic[triplet_id]: # iterate over different users/players
                word_matrix[i] = gw_datum.padded_words(max_seq_len)                
                geo_matrix[i] = np.hstack(gw_datum.geometry)
                label_matrix[i, gw_datum.label] = 1
                mask_matrix[i] = gw_datum.mask(max_seq_len)
                condition_matrix[i] = gw_datum.triplet_id
                i += 1                
                    
        return word_matrix, geo_matrix, label_matrix, mask_matrix, condition_matrix

    
class Language_Game_Data_Loader():         
    @staticmethod
    def load_dialogues_of_game(game_file):
        special_tokens = {'speaker': ' SpEaKeR: ', 'listener': ' lIsTeNeR: '}
            
        # Hard-coded incides of interesting quantities.
        triplet_id = 4
        human_role = 5       # Speaker or Listener
        sentence_pos = 6     # Linguistic data
        trial_to_data = {}

        with open(osp.join(game_file), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')

            for i, row in enumerate(reader):
                if i == 0: # skip header
                    continue
                    
                tr_id = row[triplet_id].split('_')
                
                if tr_id[0] not in ['far', 'split', 'close'] or not tr_id[1].isdigit():
                    # print 'Entry without proper triplet_id was found.'
                    continue

                tr_id = tr_id[0] + '_' + tr_id[1]
                role = special_tokens[row[human_role]]
                sentence = row[sentence_pos]

                if tr_id in trial_to_data:
                    trial_to_data[tr_id]['dialogue'] += role + sentence
                else:
                    trial_to_data[tr_id] = dict()
                    trial_to_data[tr_id]['dialogue'] = role + sentence
            return trial_to_data
        
    @staticmethod
    def load_selections_of_game(game_file):
        with open(osp.join(top_in_dir, game_file), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:                    
                    continue
                print row[15]
    
    @staticmethod
    def _triplet_to_model_map(map_file):
        res = dict()
        with open (map_file, 'r') as fin:
            for line in fin:
                tokens = line.split()
                res[tokens[0]] = (tokens[1], tokens[2], tokens[3])
            return res

    @staticmethod
    def triplet_id_to_shape_net_model_ids(close_map_file, far_map_file, split_map_file):
        '''Map from triplet_id to shape-net model ids
        '''
        triplet_to_model_map = {
        'close': Language_Game_Data_Loader._triplet_to_model_map(close_map_file),
        'far': Language_Game_Data_Loader._triplet_to_model_map(far_map_file),
        'split': Language_Game_Data_Loader._triplet_to_model_map(split_map_file)}
        return triplet_to_model_map

    
class GeoWordsDataSet(object):
    def __init__(self, words, geometries, labels, masks, conditions):
        self.num_examples = words.shape[0]
        assert(self.num_examples == labels.shape[0])
        assert(self.num_examples == geometries.shape[0])
        
        self.words = words.copy()        
        self.geometries = geometries.copy()
        self.labels = labels.copy()
        self.masks = masks.copy()
        self.conditions = conditions.copy()
        
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.words = self.words[perm]
        self.geometries = self.geometries[perm]
        self.labels = self.labels[perm]
        self.masks = self.masks[perm]
        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.words[start:end], self.geometries[start:end], self.labels[start:end], self.masks[start:end], self.conditions[start:end]
    
    
def load_glove_pretrained_model(glove_file):
    print "Loading glove model."
    embedding = dict()
    with open(glove_file, 'r') as f_in:
        for line in f_in:
            s_line = line.split()
            word = s_line[0]
            w_embedding = np.array([float(val) for val in s_line[1:]], dtype=np.float32)
            embedding[word] = w_embedding
    print "Done.", len(embedding), " words loaded!"
    return embedding