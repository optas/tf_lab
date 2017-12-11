'''
Created on August 29, 2017

@author: optas
'''

import numpy as np
import string


def _all_tensors_have_same_rows(tensor_list):
    n = tensor_list[0].shape[0]
    for i in xrange(1, len(tensor_list)):
        if n != tensor_list[i].shape[0]:
            return False
    return True

num2alpha = dict(enumerate(string.ascii_lowercase, 0))


class NumpyDataset(object):

    def __init__(self, tensor_list, tensor_names=None, copy=True, init_shuffle=True):
        '''
        Constructor
        '''
        if tensor_names is not None and len(tensor_names) != len(tensor_list):
            raise ValueError()

        if not _all_tensors_have_same_rows(tensor_list):
            raise ValueError()

        if len(tensor_list) > len(num2alpha):
            raise ValueError()

        self.n_tensors = len(tensor_list)
        self.n_examples = tensor_list[0].shape[0]

        if tensor_names is None:
            tensor_names = [num2alpha[i] for i in range(self.n_tensors)]

        print tensor_names

        self.tensor_names = tensor_names

        for name, val in zip(self.tensor_names, tensor_list):
            if copy:
                self.__setattr__(name, val.copy())
            else:
                self.__setattr__(name, val)

        self.epochs_completed = 0
        self._index_in_epoch = 0

        if init_shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.n_examples)
        np.random.shuffle(perm)

        for name in self.tensor_names:
            self.__setattr__(name, self.__getattribute__(name)[perm])
        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.n_examples
        end = self._index_in_epoch

        ret_res = []
        for name in self.tensor_names:
            ret_res.append(self.__getattribute__(name)[start:end])

        if self.n_tensors == 1:
            ret_res = ret_res[0]

        return ret_res

    def is_equal(self, other_dataset):

        if other_dataset.n_examples != self.n_examples or \
           other_dataset.n_tensors != self.n_tensors or \
           np.all(other_dataset.tensor_names != self.tensor_names):
            return False

        for name in self.tensor_names:
            if not np.all(sorted(self.__getattribute__(name)) == sorted(other_dataset.__getattribute__(name))):
                return False
        return True

    def all_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.n_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)

        ret_res = []
        for name in(self.tensor_names):
            ret_res.append(self.__getattribute__(name)[perm])
        return ret_res

## VERSION WITH OrderedDict.     
#     
# '''
# Created on Aug 29, 2017
# 
# @author: optas
# '''
# 
# import numpy as np
# import string
# from collections import OrderedDict
# 
# def _all_tensors_have_same_rows(tensor_list):
#     n = tensor_list[0].shape[0]
#     for i in xrange(1, len(tensor_list)):
#         if n != tensor_list[i].shape[0]:
#             return False
#     return True
# 
# num2alpha = dict(enumerate(string.ascii_lowercase, 0))
# 
# 
# class NumpyDataset(object):
# 
#     def __init__(self, ordered_tensors, copy=True, init_shuffle=True):
#         '''
#         Constructor.
#         '''
#         
#         is_ordered_dict =  type(ordered_tensors) is collections.OrderedDict
#         
#         if len(ordered_tensors) > len(num2alpha) and not is_ordered_dict:
#             raise ValueError()
#             
#         if not is_ordered_dict:
#             ordered_tensors_ = OrderedDict()
#             for i, t in enumerate(ordered_tensors):
#                 ordered_tensors_[num2alpha[i]] = t
#         
#             ordered_tensors = ordered_tensors_
#     
#         if not _all_tensors_have_same_rows(ordered_tensors.values()):
#             raise ValueError()
# 
#         self.tensor_names = ordered_tensors.keys()
#         self.n_tensors = len(ordered_tensors)
#         self.n_examples = ordered_tensors[self.tensor_names[0]].shape[0]
# 
#         for name, val in ordered_tensors.iteritems():
#             if copy:
#                 self.__setattr__(name, val.copy())
#             else:
#                 self.__setattr__(name, val)
# 
#         self.epochs_completed = 0
#         self._index_in_epoch = 0
# 
#         if init_shuffle:
#             self.shuffle_data()
# 
#     def shuffle_data(self, seed=None):
#         if seed is not None:
#             np.random.seed(seed)
#         perm = np.arange(self.n_examples)
#         np.random.shuffle(perm)
# 
#         for name in self.tensor_names:
#             self.__setattr__(name, self.__getattribute__(name)[perm])
#         return self
# 
#     def next_batch(self, batch_size, seed=None):
#         '''Return the next batch_size examples from this data set.
#         '''
#         start = self._index_in_epoch
#         self._index_in_epoch += batch_size
#         if self._index_in_epoch > self.n_examples:
#             self.epochs_completed += 1  # Finished epoch.
#             self.shuffle_data(seed)
#             # Start next epoch
#             start = 0
#             self._index_in_epoch = batch_size
#             assert batch_size <= self.n_examples
#         end = self._index_in_epoch
# 
#         ret_res = []
#         for name in self.tensor_names:
#             ret_res.append(self.__getattribute__(name)[start:end])
# 
#         return ret_res
# 
#     def is_equal(self, other_dataset):
# 
#         if other_dataset.n_examples != self.n_examples or \
#            other_dataset.n_tensors != self.n_tensors or \
#            np.all(other_dataset.tensor_names != self.tensor_names):
#             return False
# 
#         for name in self.tensor_names:
#             if not np.all(sorted(self.__getattribute__(name)) == sorted(other_dataset.__getattribute__(name))):
#                 return False
#         return True
# 
#     def all_data(self, shuffle=True, seed=None):
#         '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
#         '''
#         if shuffle and seed is not None:
#             np.random.seed(seed)
#         perm = np.arange(self.n_examples)  # Shuffle the data.
#         if shuffle:
#             np.random.shuffle(perm)
# 
#         ret_res = OrderedDict()
#         for name in(self.tensor_names):
#             ret_res[name] = self.__getattribute__(name)[perm]
#         return ret_res


