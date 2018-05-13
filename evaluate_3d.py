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

def evaluate_3d(model, dataloader, data_size, batch_size, phase):
    model.eval()
    running_loss = 0
    running_dice_score_class_0 = 0
    running_dice_score_class_1 = 0
    running_dice_score_class_2 = 0
    phase = phase
    
    for i,data in enumerate(dataloader[phase]):
        input, target,_ = data
        input = Variable(input).cuda()
        true = Variable(torch.transpose(target,2,1).contiguous().view(-1,4,256,256)).cuda()
        output = model(input)
        output_reshaped = torch.transpose(output,2,1).contiguous().view(-1,4,256,256)
        loss = dice_loss(true,output_reshaped)
        running_loss += loss.data[0] * batch_size
        dice_score_batch = dice_score(true,output_reshaped)
        running_dice_score_class_0 += dice_score_batch[0] * batch_size
        running_dice_score_class_1 += dice_score_batch[1] * batch_size
        running_dice_score_class_2 += dice_score_batch[2] * batch_size
        preds = predict(output_reshaped)
        if i == 4:
            for k in range(15):
                for j in range(3):
                    print('True Map')
                    plt.imshow(input[0,1,k,:,:].data.cpu().numpy())
                    plt.show()
                    plt.imshow(target[0,j,k,:,:].cpu().numpy())
                    plt.show()
                    print('Predicted Map')
                    plt.imshow(preds[j][k,:,:].cpu().numpy())
                    plt.show()
            
    loss = running_loss/data_sizes[phase] 
    dice_score_0 = running_dice_score_class_0/data_sizes[phase]
    dice_score_1 = running_dice_score_class_1/data_sizes[phase]
    dice_score_2 = running_dice_score_class_2/data_sizes[phase] 
    print('{} loss: {:.4f}, Dice Score (class 0): {:.4f}, Dice Score (class 1): {:.4f},Dice Score (class 2): {:.4f}'.format(phase,loss, dice_score_0, dice_score_1, dice_score_2))
    return loss, dice_score_0, dice_score_1, dice_score_2

