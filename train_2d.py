import time
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

from utils import *

def train_model(model, optimizer,dataloader, data_sizes, batch_size, name, num_epochs = 100,
                verbose = False, dice_loss = dice_loss, noisy_labels = False):
    since = time.time()
    best_loss = np.inf
    best_dice_cl0 = 0
    best_dice_cl1 = 0
    best_dice_cl2 = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_hist = {'train':[],'validate':[]}
    dice_score_0_hist = {'train':[],'validate':[]}
    dice_score_1_hist = {'train':[],'validate':[]}
    dice_score_2_hist = {'train':[],'validate':[]}
    for i in range(num_epochs):
        for phase in ['train', 'validate']:
            running_loss = 0
            running_dice_score_class_0 = 0
            running_dice_score_class_1 = 0
            running_dice_score_class_2 = 0
            
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
    
            for data in dataloader[phase]:
                optimizer.zero_grad()
                input, segF, segP, segT,_ = data
                input = Variable(input).cuda()
                true = Variable(segments(segF, segP, segT)).cuda()
                output = model(input)
                if noisy_labels and phase == 'train':
                    noise = generate_noise(true)
                    true = true + true*noise
                if noisy_labels:
                    loss = entropy_loss(true,output)
                else:
                    loss = dice_loss(true,output)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0] * batch_size
                dice_score_batch = dice_score(true,output)
                running_dice_score_class_0 += dice_score_batch[0] * batch_size
                running_dice_score_class_1 += dice_score_batch[1] * batch_size
                running_dice_score_class_2 += dice_score_batch[2] * batch_size
            epoch_loss = running_loss/data_sizes[phase]
            loss_hist[phase].append(epoch_loss) 
            epoch_dice_score_0 = running_dice_score_class_0/data_sizes[phase]
            epoch_dice_score_1 = running_dice_score_class_1/data_sizes[phase]
            epoch_dice_score_2 = running_dice_score_class_2/data_sizes[phase]
            dice_score_0_hist[phase].append(epoch_dice_score_0)
            dice_score_1_hist[phase].append(epoch_dice_score_1)
            dice_score_2_hist[phase].append(epoch_dice_score_2)
            if verbose or i%10 == 0:
                print('Epoch: {}, Phase: {}, epoch loss: {:.4f}, Dice Score (class 0): {:.4f}, Dice Score (class 1): {:.4f},Dice Score (class 2): {:.4f}'.format(i,phase,epoch_loss, epoch_dice_score_0, epoch_dice_score_1, epoch_dice_score_2))
                print('-'*10)
            
        if phase == 'validate' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model,name)
            best_dice_cl0 = epoch_dice_score_0
            best_dice_cl1 = epoch_dice_score_1
            best_dice_cl2 = epoch_dice_score_2
    print('-'*50)    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val dice loss: {:4f}, dice score (class 0): {:.4f}, dice score (class 1): {:.4f},dice score (class 2): {:.4f}'\
          .format(best_loss, best_dice_cl0, best_dice_cl1, best_dice_cl2))
    
    model.load_state_dict(best_model_wts)
    
    return model, loss_hist, dice_score_0_hist, dice_score_1_hist, dice_score_2_hist

import time
def train_model_patches(model, optimizer,dataloader, data_sizes, batch_size, num_epochs = 100, verbose = False):
    since = time.time()
    best_loss = np.inf
    loss_hist = {'train':[],'validate':[]}
    dice_score_0_hist = {'train':[],'validate':[]}
    dice_score_1_hist = {'train':[],'validate':[]}
    dice_score_2_hist = {'train':[],'validate':[]}
    for i in range(num_epochs):
        for phase in ['train', 'validate']:
            running_loss = 0
            running_dice_score_class_0 = 0
            running_dice_score_class_1 = 0
            running_dice_score_class_2 = 0
            
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
    
            for data in dataloader[phase]:
                optimizer.zero_grad()
                input, seg,_ = data
                input = Variable(input[:,:,1:49,1:49]).cuda()
                true = Variable(seg[:,:,1:49,1:49]).cuda()
                output = model(input)
                loss = dice_loss(true,output)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0] * batch_size
                dice_score_batch = dice_score(true,output)
                running_dice_score_class_0 += dice_score_batch[0] * batch_size
                running_dice_score_class_1 += dice_score_batch[1] * batch_size
                running_dice_score_class_2 += dice_score_batch[2] * batch_size
            epoch_loss = running_loss/data_sizes[phase]
            loss_hist[phase].append(epoch_loss) 
            epoch_dice_score_0 = running_dice_score_class_0/data_sizes[phase]
            epoch_dice_score_1 = running_dice_score_class_1/data_sizes[phase]
            epoch_dice_score_2 = running_dice_score_class_2/data_sizes[phase]
            dice_score_0_hist[phase].append(epoch_dice_score_0)
            dice_score_1_hist[phase].append(epoch_dice_score_1)
            dice_score_2_hist[phase].append(epoch_dice_score_2)
            if verbose or i%10 == 0:
                print('Epoch: {}, Phase: {}, epoch loss: {:.4f}, Dice Score (class 1): {:.4f}, Dice Score (class 2): {:.4f},Dice Score (class 3): {:.4f}'.format(i,phase,epoch_loss, epoch_dice_score_0, epoch_dice_score_1, epoch_dice_score_2))
                print('-'*10)
            
        if phase == 'validate' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict() 
        
        if phase == 'validate':
            torch.save(model,'unet_patches_4_epoch{}'.format(i))
            
    print('-'*50)    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val dice loss: {:4f}'.format(best_loss))
    
    model.load_state_dict(best_model_wts)
    
    return model, loss_hist, dice_score_0_hist, dice_score_1_hist, dice_score_2_hist