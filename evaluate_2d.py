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

def evaluate(model, dataloader, data_size, batch_size, phase, dice_loss = dice_loss, noisy_labels = False):
    model.train(False)
    running_loss = 0
    running_dice_score_class_0 = 0
    running_dice_score_class_1 = 0
    running_dice_score_class_2 = 0
    phase = phase
    
    for i,data in enumerate(dataloader[phase]):
        input_1, segF,segP, segT,_ = data
        input_1 = Variable(input_1).cuda()
        output = model(input_1)
        true = Variable(segments(segF, segP, segT)).cuda()
        if noisy_labels:
            loss = entropy_loss(true,output)
        else:
            loss = dice_loss(true,output)
        running_loss += loss.data[0] * batch_size
        dice_score_batch = dice_score(true,output)
        running_dice_score_class_0 += dice_score_batch[0] * batch_size
        running_dice_score_class_1 += dice_score_batch[1] * batch_size
        running_dice_score_class_2 += dice_score_batch[2] * batch_size
        preds = predict(output)
        if i == 11 or i == 4:
            for k in range(batch_size):
                for j in range(3):
                    print('True Map')
                    plt.imshow(input_1[k,1,:,:].data.cpu().numpy())
                    plt.show()
                    plt.imshow(true[k,j,:,:].data.cpu().numpy())
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

def evaluate_exp(model, dataloader, data_size, batch_size, phase, dice_loss = dice_loss):
    model.train(False)
    running_loss = 0
    running_dice_score_class_0 = 0
    running_dice_score_class_1 = 0
    running_dice_score_class_2 = 0
    phase = phase
    
    for i,data in enumerate(dataloader[phase]):
        input_1, segF,segP, segT,_ = data
        input_1 = Variable(input_1).cuda()
        input_1[:,:7,:,:] = input_1[:, [6,5,4,3,2,1,0], :, :]
        output = model(input_1)
        true = Variable(segments(segF, segP, segT)).cuda()
        loss = dice_loss(true,output)
        running_loss += loss.data[0] * batch_size
        dice_score_batch = dice_score(true,output)
        running_dice_score_class_0 += dice_score_batch[0] * batch_size
        running_dice_score_class_1 += dice_score_batch[1] * batch_size
        running_dice_score_class_2 += dice_score_batch[2] * batch_size
        preds = predict(output)
        if i == 11 or i == 4:
            for k in range(batch_size):
                for j in range(3):
                    print('True Map')
                    plt.imshow(input_1[k,1,:,:].data.cpu().numpy())
                    plt.show()
                    plt.imshow(true[k,j,:,:].data.cpu().numpy())
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

def evaluate_patches(model, dataloader, data_size, batch_size, phase, patch = True):
    model.eval()
    running_loss = 0
    running_dice_score_class_0 = 0
    running_dice_score_class_1 = 0
    running_dice_score_class_2 = 0
    phase = phase
    
    for i,data in enumerate(dataloader[phase]):
        input, seg,_ = data
        if patch:
            input = Variable(input[:,:,1:49,1:49]).cuda()
            true = Variable(seg[:,:,1:49,1:49]).cuda()
        else:
            input = Variable(input).cuda()
            true = Variable(seg).cuda()
        output = model(input)
        loss = dice_loss(true,output)
        running_loss += loss.data[0] * batch_size
        dice_score_batch = dice_score(true,output)
        running_dice_score_class_0 += dice_score_batch[0] * batch_size
        running_dice_score_class_1 += dice_score_batch[1] * batch_size
        running_dice_score_class_2 += dice_score_batch[2] * batch_size
        preds = predict(output)
        if i == 11:
            for k in range(batch_size):
                for j in range(3):
                    print('True Map')
                    plt.imshow(input[k,1,:,:].data.cpu().numpy())
                    plt.show()
                    plt.imshow(true[k,j+1,:,:].data.cpu().numpy())
                    plt.show()
                    print('Predicted Map')
                    plt.imshow(preds[j][k,:,:].cpu().numpy())
                    plt.show()
            
    loss = running_loss/data_sizes[phase] 
    dice_score_0 = running_dice_score_class_0/data_sizes[phase]
    dice_score_1 = running_dice_score_class_1/data_sizes[phase]
    dice_score_2 = running_dice_score_class_2/data_sizes[phase] 
    print('{} loss: {:.4f}, Dice Score (class 1): {:.4f}, Dice Score (class 2): {:.4f},Dice Score (class 3): {:.4f}'.format(phase,loss, dice_score_0, dice_score_1, dice_score_2))
    return loss, dice_score_0, dice_score_1, dice_score_2

