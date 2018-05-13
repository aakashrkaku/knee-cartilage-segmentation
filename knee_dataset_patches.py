import os
import os.path
import numpy as np
import random
import h5py
import torch
import glob
import torch.utils.data as udata


class MRI_Dataset(udata.Dataset):
    def __init__(self, x_path, y_path):
        super(MRI_Dataset, self).__init__()

        self.x_path = x_path;
        self.y_path = y_path;
  
        h5f_x = h5py.File(self.x_path, 'r')
        h5f_y = h5py.File(self.y_path, 'r')

        self.keys = list(h5f_x.keys())
        random.shuffle(self.keys)
        h5f_x.close()
        h5f_y.close()

    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):

        h5f_x = h5py.File(self.x_path, 'r')
        h5f_y = h5py.File(self.y_path, 'r')

        key = self.keys[index]
        data_x = np.array(h5f_x[key])
        data_y = np.array(h5f_y[key])

        h5f_x.close();
        h5f_y.close();
        return (torch.Tensor(data_x), torch.Tensor(data_y),key)
