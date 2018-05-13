import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage import io, transform
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import scipy
import random
import pickle
import scipy.io as sio
import itertools
from scipy.ndimage.interpolation import shift

import warnings
warnings.filterwarnings("ignore")
plt.ion()

class KneeMRI3DDataset(Dataset):
    '''Knee MRI Dataset'''
    def __init__(self, root_dir, label, train_data = False, flipping = True, rotation = True, translation = True,
                normalize = False):
        self.root_dir = root_dir
        self.label = label
        self.flipping = flipping
        self.rotation = rotation
        self.translation = translation
        self.train_data = train_data
        self.normalize = normalize
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        variable_path_name = os.path.join(self.root_dir, self.label[idx])
        variables = sio.loadmat(variable_path_name)
        segment_T = variables['SegmentationT'].transpose(2,0,1).astype(float)
        segment_F = variables['SegmentationF'].transpose(2,0,1).astype(float)
        segment_P = variables['SegmentationP'].transpose(2,0,1).astype(float)
        images = []
        md = variables['MDnr'].transpose(2,0,1)
        image = variables['NUFnr'].transpose(3,2,0,1)
        fa = variables['FAnr'].transpose(2,0,1)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        fa = torch.from_numpy(fa).type(torch.FloatTensor)
        md = torch.from_numpy(md).type(torch.FloatTensor)
        images.append(image)
        images.append(fa.unsqueeze(0))
        images.append(md.unsqueeze(0))
        image_all = torch.cat(images, dim = 0)
        segment_F = torch.from_numpy(segment_F).type(torch.FloatTensor)
        segment_T = torch.from_numpy(segment_T).type(torch.FloatTensor)
        segment_P = torch.from_numpy(segment_P).type(torch.FloatTensor)
        segments = []
        segments.append(segment_F.unsqueeze(0))
        segments.append(segment_T.unsqueeze(0))
        segments.append(segment_P.unsqueeze(0))
        seg_tot = segment_F + segment_T + segment_P
        seg_none = (seg_tot == 0).type(torch.FloatTensor)
        segments.append(seg_none.unsqueeze(0))
        segments_all = torch.cat(segments, dim = 0)
        
        if self.normalize:
            max_image, min_image, max_fa, min_fa, max_md, min_md = pickle.load(open('normalizing_values_new','rb'))
            image_all[:7,:,:,:] = (image_all[:7,:,:,:] - min_image)/(max_image - min_image)
            image_all[7,:,:,:] = (image_all[7,:,:,:] - min_fa)/(max_fa - min_fa)
            image_all[-1,:,:,:] = (image_all[-1,:,:,:] - min_md)/(max_md - min_md)
        
        return (image_all,segments_all,self.label[idx])
