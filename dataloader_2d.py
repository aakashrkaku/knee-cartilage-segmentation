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
import copy
import warnings
warnings.filterwarnings("ignore")
plt.ion()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class KneeMRIDataset(Dataset):
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
        variables = pickle.load(open(variable_path_name,'rb'))
        segment_T = variables['SegmentationT'].astype(float)
        segment_F = variables['SegmentationF'].astype(float)
        segment_P = variables['SegmentationP'].astype(float)
        md = variables['MDnr']
        image = variables['NUFnr']
        fa = variables['FAnr']
        images = []
        flip = random.random() > 0.5
        angle = random.uniform(-4,4)
        dx = np.round(random.uniform(-7,7))
        dy = np.round(random.uniform(-7,7))
        for i in range(7):
            im = image[:,:,i]
            if self.train_data:
                if self.flipping and flip:
                    im = np.fliplr(im)
                if self.rotation:
                    im = transform.rotate(im, angle, order = 0)
                if self.translation:
                    im = shift(im,(dx,dy), order = 0)
            im = torch.from_numpy(im).type(torch.DoubleTensor)
            images.append(im.unsqueeze(0))
        if self.train_data:
            if self.flipping and flip:
                segment_T = np.fliplr(segment_T)
                segment_F = np.fliplr(segment_F)
                segment_P = np.fliplr(segment_P)
                md = np.fliplr(md)
                fa = np.fliplr(fa)
            if self.rotation:
                segment_T = transform.rotate(segment_T,angle, order = 0)
                segment_F = transform.rotate(segment_F,angle, order = 0)
                segment_P = transform.rotate(segment_P,angle, order = 0)
                md = transform.rotate(md,angle, order = 0)
                fa = transform.rotate(fa,angle, order = 0)
            if self.translation:
                segment_T = shift(segment_T,(dx,dy), order = 0)
                segment_F = shift(segment_F,(dx,dy), order = 0)
                segment_P = shift(segment_P,(dx,dy), order = 0)
                md = shift(md,(dx,dy), order = 0)
                fa = shift(fa,(dx,dy), order = 0)
        fa = torch.from_numpy(fa).type(torch.DoubleTensor)
        md = torch.from_numpy(md).type(torch.DoubleTensor)
        images.append(fa.unsqueeze(0))
        images.append(md.unsqueeze(0))
        image_all = torch.cat(images, dim = 0)
        segment_F = torch.from_numpy(segment_F)
        segment_T = torch.from_numpy(segment_T)
        segment_P = torch.from_numpy(segment_P)
        if self.normalize:
#             max_image, min_image, max_fa, min_fa, max_md, min_md = pickle.load(open('normalizing_values','rb'))
            max_image, min_image = torch.max(image_all[:7,:,:]), torch.min(image_all[:7,:,:])
            max_fa, min_fa = torch.max(image_all[7,:,:]), torch.min(image_all[7,:,:])
            max_md, min_md = torch.max(image_all[8,:,:]), torch.min(image_all[8,:,:])
            image_all[:7,:,:] = (image_all[:7,:,:] - min_image)/(max_image - min_image)
            image_all[7,:,:] = (image_all[7,:,:] - min_fa)/(max_fa - min_fa)
            image_all[-1,:,:] = (image_all[-1,:,:] - min_md)/(max_md - min_md)
        
        return (image_all.type(torch.FloatTensor), segment_F.type(torch.FloatTensor), 
                segment_P.type(torch.FloatTensor), segment_T.type(torch.FloatTensor),self.label[idx])

