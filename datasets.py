import os, sys, random
import numpy as np 
import scipy.io as sio

import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import utils

def load_data(params, split=True):
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []

    
    if params['data_name'] in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(params['main_dir'], 'data', params['data_name'] +'.mat'))

        x1 = mat['X1']
        x1 = utils.normalize(x1).astype('float32')
        y1 = np.squeeze(mat['Y']).astype('int')
        
        x2 = mat['X2']
        x2 = utils.normalize(x1).astype('float32')
        y2 = np.squeeze(mat['Y']).astype('int')
        
        X_list = [x1, x2]
        Y_list = [y1, y2]

    if split:
        ## shuffle data
        index = random.sample(range(X_list[0].shape[0]), X_list[0].shape[0])
        half_index = int(len(index)*params['aligned_ratio'])

        for view in range(params['view_size']):
            train_index = index[:half_index]
            test_index = index[half_index:]

            train_X_list.append(X_list[view][train_index])
            train_Y_list.append(Y_list[view][train_index])

            test_X_list.append(X_list[view][test_index])
            test_Y_list.append(Y_list[view][test_index])
        return X_list, Y_list, train_X_list, train_Y_list, test_X_list, test_Y_list
    else:
        return X_list, Y_list


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, X_list):

        self.X_list = X_list
        self.view_size = len(X_list)

    def __getitem__(self, index):
        current_x_list = []
        for view in range(self.view_size):
            current_x = self.X_list[view][index]
            current_x_list.append(current_x)
        
        # permutation
        P_index = random.sample(range(len(index)), len(index))
        P = np.eye(len(index)).astype('float32')
        P = P[:, P_index]
        current_x_list[1] = current_x_list[1][P_index]
        return current_x_list, P

    def __len__(self):
        # return the total size of data
        return  self.X_list[0].shape[0]


class Data_Sampler(object):
    """Custom Sampler is required. This sampler prepares batch by passing list of
    data indices instead of running over individual index as in pytorch sampler"""
    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size