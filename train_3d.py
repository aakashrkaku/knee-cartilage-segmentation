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

import warnings
warnings.filterwarnings("ignore")
plt.ion()

from utils import *

def train_model_3d(model, optimizer,dataloader, data_sizes, batch_size, num_epochs = 100, verbose = False):
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
                input, target,_ = data
                input = Variable(input).cuda()
                true = Variable(torch.transpose(target,2,1).contiguous().view(-1,4,256,256)).cuda()
                output = model(input)
                output_reshaped = torch.transpose(output,2,1).contiguous().view(-1,4,256,256)
                loss = dice_loss(true,output_reshaped,p=2.5)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0] * batch_size
                dice_score_batch = dice_score(true,output_reshaped)
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
            best_model_wts = model.state_dict() 

    print('-'*50)    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val dice loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)

    return model, loss_hist, dice_score_0_hist, dice_score_1_hist, dice_score_2_hist