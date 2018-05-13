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


class Input_Vnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Input_Vnet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.in_channels = in_channels
    
    def forward(self,x):
        ones = Variable(torch.ones(1,self.in_channels,1,256,256)).cuda()
        x = torch.cat([x,ones], dim = 2)
        return F.relu(self.bn1(self.conv1(x)))

class Downsample_Vnet(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample = (2,2,2), num_layers = 1, drop_out = False):
        super(Downsample_Vnet,self).__init__()
        self.layer = nn.ModuleList()
        for i in range(num_layers):
            self.layer.append(nn.Conv3d(out_channels, out_channels, 3, padding =1))
            self.layer.append(nn.BatchNorm3d(out_channels))
            self.layer.append(nn.ReLU())
        self.down_conv = nn.Conv3d(in_channels, out_channels, down_sample, stride = down_sample)
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.drop_out = drop_out
    
    def forward(self, x):
        a = F.relu(self.down_bn(self.down_conv(x)))
        if self.drop_out:
            x = F.dropout3d(a)
        else:
            x = a
        for i in self.layer:
            x = i(x)
        x = F.relu(torch.add(x,a))
        
        return x

class Upsample_Vnet(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample = (2,2,2), num_layers = 1, drop_out = False):
        super(Upsample_Vnet, self).__init__()
        self.layer = nn.ModuleList()
        for i in range(num_layers):
            self.layer.append(nn.Conv3d(out_channels, out_channels, 3, padding =1))
            self.layer.append(nn.BatchNorm3d(out_channels))
            self.layer.append(nn.ReLU())
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels//2, up_sample, stride = up_sample)
        self.up_bn = nn.BatchNorm3d(out_channels//2)
        self.drop_out = drop_out
        
    def forward(self, x, y):
        if self.drop_out:
            x = F.dropout3d(x)
        x = F.relu(self.up_bn(self.up_conv(x)))
        y = F.dropout3d(y)
        x = torch.cat([x,y], dim = 1)
        a = x
        for i in self.layer:
            x = i(x)
        x = F.relu(torch.add(x,a))
        
        return x


class Output_Vnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Output_Vnet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x[:,:,:15,:,:]


class Vnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Vnet, self).__init__()
        self.input_layer = Input_Vnet(in_channels, 16)
        self.down1 = Downsample_Vnet(16, 32, down_sample=(1,2,2), num_layers=2)
        self.down2 = Downsample_Vnet(32, 64, num_layers= 3)
        self.down3 = Downsample_Vnet(64, 128, down_sample = (1,2,2),num_layers = 3, drop_out=True)
        self.down4 = Downsample_Vnet(128, 256, num_layers= 3, drop_out=True)
        self.up4 = Upsample_Vnet(256, 256, num_layers = 3, drop_out=True)
        self.up3 = Upsample_Vnet(256,128, up_sample= (1,2,2), num_layers = 3, drop_out=True)
        self.up2 = Upsample_Vnet(128,64, num_layers = 2)
        self.up1 = Upsample_Vnet(64, 32, up_sample=(1,2,2), num_layers=1)
        self.output = Output_Vnet(32, 4)
    
    def forward(self, x):
        x_in = self.input_layer(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)
        x = self.up4(x_d4, x_d3)
        x = self.up3(x,x_d2)
        x = self.up2(x, x_d1)
        x = self.up1(x, x_in)
        x = self.output(x)
        
        return x
