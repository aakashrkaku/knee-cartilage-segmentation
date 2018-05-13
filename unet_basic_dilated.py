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

class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2,stride = 2)
        
        return x, y


class Down_sample_dilated_block(nn.Module):
    def __init__(self,in_channels, out_channels, dropout = False):
        super(Down_sample_dilated_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding= 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2_1d = nn.Conv2d(out_channels, out_channels, 3, padding= 1)
        self.bn2_1d = nn.BatchNorm2d(out_channels)
        self.conv2_2d = nn.Conv2d(out_channels, out_channels, 3, padding = 3, dilation = 3)
        self.bn2_2d = nn.BatchNorm2d(out_channels)
        self.conv2_3d = nn.Conv2d(out_channels, out_channels, 3, padding = 5, dilation = 5)
        self.bn2_3d = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels *3, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv_down = nn.Conv2d(out_channels, out_channels, 2, stride = 2)
        self.bn_down_conv = nn.BatchNorm2d(out_channels)
        self.dp = dropout
    
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2_1d(self.conv2_1d(x)))
        x2 = F.relu(self.bn2_2d(self.conv2_2d(x)))
        x3 = F.relu(self.bn2_3d(self.conv2_3d(x)))
        x = torch.cat([x1,x2,x3], dim = 1)
        y = F.relu(self.bn3(self.conv3(x)))
        if self.dp:
            x = F.dropout2d(F.relu(self.bn_down_conv(self.conv_down(y))))
        else:
            x = F.relu(self.bn_down_conv(self.conv_down(y)))
        
        return x, y


class Upsample_block(nn.Module):
    def __init__(self,in_channels, out_channels, dropout = False):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding = 1, stride = 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dp = dropout
        
    def forward(self,x, y):
        x = self.transconv(x)
        if self.dp:
            y = F.dropout2d(y)
        x = torch.cat((x,y),dim = 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x

class Upsample_block_small(nn.Module):
    def __init__(self,in_channels, out_channels, dropout = False):
        super(Upsample_block_small, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding = 1, stride = 2)
        self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dp = dropout
        
    def forward(self,x, y):
        x = self.transconv(x)
        if self.dp:
            y = F.dropout2d(y)
        x = torch.cat((x,y),dim = 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x

class Unet_dilated_gen(nn.Module):
    def __init__(self, in_layers, out_layers, dilated = True, dropout = False):
        super(Unet_dilated_gen, self).__init__()
        if dilated:
            self.down1 = Down_sample_dilated_block(in_layers,64)
            self.down2 = Down_sample_dilated_block(64,128)
            self.down3 = Down_sample_dilated_block(128,256,dropout=dropout)
            self.down4 = Down_sample_dilated_block(256,512,dropout=dropout)
        else:
            self.down1 = Downsample_block(in_layers,64)
            self.down2 = Downsample_block(64,128)
            self.down3 = Downsample_block(128,256)
            self.down4 = Downsample_block(256,512)
        self.conv1 = nn.Conv2d(512,1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024,1024,3, padding = 1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.up4 = Upsample_block(1024,512)
        self.up3 = Upsample_block(512,256)
        self.up2 = Upsample_block(256,128, dropout=dropout)
        self.up1 = Upsample_block(128,64, dropout=dropout)
        self.outconv = nn.Conv2d(64,out_layers, 1)
        
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


class Unet_dilated_small_deep(nn.Module):
    def __init__(self, in_layers, out_layers, dilated = True, int_var = 32):
        super(Unet_dilated_small_deep, self).__init__()
        if dilated:
            self.down1 = Down_sample_dilated_block(in_layers,int_var)
            self.down2 = Down_sample_dilated_block(int_var,int_var)
            self.down3 = Down_sample_dilated_block(int_var,int_var)
            self.down4 = Down_sample_dilated_block(int_var,int_var)
            self.down5 = Down_sample_dilated_block(int_var,int_var)
            self.down6 = Down_sample_dilated_block(int_var,int_var)
        else:
            self.down1 = Downsample_block(in_layers,int_var)
            self.down2 = Downsample_block(int_var,int_var)
            self.down3 = Downsample_block(int_var,int_var)
            self.down4 = Downsample_block(int_var,int_var)
            self.down5 = Downsample_block(int_var,int_var)
            self.down6 = Downsample_block(int_var,int_var)
        self.conv1 = nn.Conv2d(int_var,int_var, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(2048)
        self.conv2 = nn.Conv2d(int_var,int_var,3, padding = 1)
        self.bn2 = nn.BatchNorm2d(2048)
        self.up5 = Upsample_block(int_var,int_var)
        self.up4 = Upsample_block(int_var,int_var)
        self.up3 = Upsample_block(int_var,int_var)
        self.up2 = Upsample_block(int_var,int_var)
        self.up1 = Upsample_block(int_var,int_var)
        self.outconv = nn.Conv2d(int_var,out_layers, 1)
        
    def forward(self,x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x, y5 = self.down5(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up5(x, y5)
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x = self.outconv(x)
        
        return x

class Unet_dilated_small(nn.Module):
    def __init__(self, in_layers, out_layers, int_var = 64, dilated = True, dropout = False):
        super(Unet_dilated_small, self).__init__()
        if dilated:
            self.down1 = Down_sample_dilated_block(in_layers,int_var)
            self.down2 = Down_sample_dilated_block(int_var,int_var)
            self.down3 = Down_sample_dilated_block(int_var,int_var,dropout=dropout)
            self.down4 = Down_sample_dilated_block(int_var,int_var,dropout=dropout)
        else:
            self.down1 = Downsample_block(in_layers,int_var)
            self.down2 = Downsample_block(int_var,int_var)
            self.down3 = Downsample_block(int_var,int_var)
            self.down4 = Downsample_block(int_var,int_var)
        self.conv1 = nn.Conv2d(int_var,int_var, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(int_var)
        self.conv2 = nn.Conv2d(int_var,int_var,3, padding = 1)
        self.bn2 = nn.BatchNorm2d(int_var)
        self.up4 = Upsample_block_small(int_var,int_var)
        self.up3 = Upsample_block_small(int_var,int_var)
        self.up2 = Upsample_block_small(int_var,int_var, dropout=dropout)
        self.up1 = Upsample_block_small(int_var,int_var, dropout=dropout)
        self.outconv = nn.Conv2d(int_var,out_layers, 1)
        
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

class Unet_dilated_small_multi(nn.Module):
    def __init__(self, in_layers, out_layers, int_var = 64, dilated = True, dropout = False):
        super(Unet_dilated_small_multi, self).__init__()
        if dilated:
            self.down1 = Down_sample_dilated_block(in_layers,int_var)
            self.down2 = Down_sample_dilated_block(int_var,int_var)
            self.down3 = Down_sample_dilated_block(int_var,int_var,dropout=dropout)
            self.down4 = Down_sample_dilated_block(int_var,int_var,dropout=dropout)
        else:
            self.down1 = Downsample_block(in_layers,int_var)
            self.down2 = Downsample_block(int_var,int_var)
            self.down3 = Downsample_block(int_var,int_var)
            self.down4 = Downsample_block(int_var,int_var)
        self.conv1 = nn.Conv2d(int_var,int_var, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(int_var)
        self.conv2 = nn.Conv2d(int_var,int_var,3, padding = 1)
        self.bn2 = nn.BatchNorm2d(int_var)
        self.up_layer = nn.ModuleList()
        self.out_layers = out_layers
        for i in range(self.out_layers):
            self.up_layer.append(nn.ModuleList())
            self.up_layer[i].append(Upsample_block_small(int_var,int_var))
            self.up_layer[i].append(Upsample_block_small(int_var,int_var))
            self.up_layer[i].append(Upsample_block_small(int_var,int_var, dropout=dropout))
            self.up_layer[i].append(Upsample_block_small(int_var,int_var, dropout=dropout))
            self.up_layer[i].append(nn.Conv2d(int_var,1,1))
        
    def forward(self,x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        xs = []
        for i in range(self.out_layers):
            temp_x = self.up_layer[i][0](x,y4)
            temp_x = self.up_layer[i][1](temp_x,y3)
            temp_x = self.up_layer[i][2](temp_x,y2)
            temp_x = self.up_layer[i][3](temp_x,y1)
            xs.append(self.up_layer[i][4](temp_x))
        x = torch.cat(xs,dim=1)
        
        return x