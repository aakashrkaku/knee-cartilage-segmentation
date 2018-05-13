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

class Downsample_Unet3d(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample = (1,2,2)):
        super(Downsample_Unet3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, down_sample, stride = down_sample)
        self.bn3 = nn.BatchNorm3d(out_channels)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool3d(y, (1,2,2),stride = (1,2,2))
        x = F.relu(self.bn3(self.conv3(y)))
        
        return x, y

class Upsample_Unet3d(nn.Module):
    def __init__(self,in_channels, out_channels, up_sample = (1,2,2)):
        super(Upsample_Unet3d, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, up_sample, stride = up_sample)
        self.bnup = nn.BatchNorm3d(out_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x, y):
        x = F.relu(self.bnup(self.upconv(x)))
        x = torch.cat((x,y),dim = 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x

class Unet_3D(nn.Module):
    def __init__(self, in_layers, out_layers):
        super(Unet_3D, self).__init__()
        self.down1 = Downsample_Unet3d(in_layers,32)
        self.down2 = Downsample_Unet3d(32,64)
        self.down3 = Downsample_Unet3d(64,128)
        self.down4 = Downsample_Unet3d(128,256)
        self.conv1 = nn.Conv3d(256,512, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(512)
        self.conv2 = nn.Conv3d(512,512,3, padding = 1)
        self.bn2 = nn.BatchNorm3d(512)
        self.up4 = Upsample_Unet3d(512,256)
        self.up3 = Upsample_Unet3d(256,128)
        self.up2 = Upsample_Unet3d(128,64)
        self.up1 = Upsample_Unet3d(64,32)
        self.outconv = nn.Conv3d(32,out_layers, 1)
        
    def forward(self,x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x = self.outconv(x)
        
        return x

