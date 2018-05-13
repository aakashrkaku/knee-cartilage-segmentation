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

class PPnet(nn.Module):
    def __init__(self, in_channels, out_channels, int_chan = 16, num_layers = 4):
        super(PPnet, self).__init__()
        self.layers = nn.ModuleList()
        in_chan = in_channels
        for i in range(num_layers):
            if i == num_layers-1:
                self.layers.append(nn.Conv2d(int_chan,out_channels,3,padding = 1))
            else:
                self.layers.append(nn.Conv2d(in_chan, int_chan, 3, padding= 1))
                self.layers.append(nn.BatchNorm2d(int_chan))
                self.layers.append(nn.ReLU())
                in_chan = int_chan
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self,x):
        return self.layers(x)

def predict_pp(scores):
    preds = F.softmax(scores)
    pred_class = torch.max(preds, dim = 1, keepdim = True)[1]
    class_0_pred_seg = (pred_class == 0).type(torch.cuda.FloatTensor)
    class_1_pred_seg = (pred_class == 1).type(torch.cuda.FloatTensor)
    class_2_pred_seg = (pred_class == 2).type(torch.cuda.FloatTensor)
    class_3_pred_seg = (pred_class == 3).type(torch.cuda.FloatTensor)
    prediction = torch.cat((class_0_pred_seg,class_1_pred_seg,class_2_pred_seg,class_3_pred_seg),dim = 1)
    return prediction

def smoothing_preds(preds,filter_size = 3):
    smooth_pred = torch.round(F.avg_pool1d(preds,filter_size))
    return smooth_pred

from sklearn.metrics import accuracy_score
def dice_score_list(true,scores):
    N,C,sh1,sh2 = true.size()
    prediction = predict_pp(scores)
    prediction = prediction.data.cpu().numpy()
    true = true.data.cpu().numpy()
    dc_sr = {0:[],1:[],2:[]}
    acc_sr = {0:[],1:[],2:[]}
    for i in range(N):
        for j in range(3):
            pred = prediction[i,j,:,:]
            tr = true[i,j,:,:]
            flat_pred = np.ravel(pred)
            flat_true = np.ravel(tr)
            acc_sr[j].append(accuracy_score(flat_true,flat_pred))
            temp_dice_F = (2*(np.sum(pred*tr)) + 1e-4)/(np.sum(pred + tr) + 1e-4)
            dc_sr[j].append(temp_dice_F)
    return dc_sr, acc_sr


