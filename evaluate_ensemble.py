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


def evaluate_pp(model,prediction_models, dataloader, data_size, batch_size, phase, dice_loss = dice_loss,\
                smooth = False, filter_size = 3, print_all = False, certainity_map = False):
    model.eval()
    running_loss = 0
    running_dice_score_class_0 = 0
    running_dice_score_class_1 = 0
    running_dice_score_class_2 = 0
    dc_sr = {0:[],1:[],2:[]}
    acc_sr = {0:[],1:[],2:[]}
    phase = phase
    for i in prediction_models:
        for param in i.parameters():
            param.requires_grad = False
    
    for i,data in enumerate(dataloader[phase]):
        input, segF,segP, segT,_ = data
        input = Variable(input).cuda()
        input_pp = []
        for j in prediction_models:
            output = j(input)
            preds_m = predict_pp(output)
            input_pp.append(preds_m)
        input_pp = torch.cat(input_pp,dim = 1)
        output_pp = model(input_pp)
        true = Variable(segments(segF, segP, segT)).cuda()
        loss = dice_loss(true,output_pp)
        running_loss += loss.data[0] * batch_size
        dice_score_batch = dice_score(true,output_pp, smooth= smooth, filter_size=filter_size)
        running_dice_score_class_0 += dice_score_batch[0] * batch_size
        running_dice_score_class_1 += dice_score_batch[1] * batch_size
        running_dice_score_class_2 += dice_score_batch[2] * batch_size
        dc_dict, acc_dict = dice_score_list(true,output_pp)
        if certainity_map:
            cm = make_certainity_maps(output_pp)
        for k in range(3):
            dc_sr[k].append(dc_dict[k])
            acc_sr[k].append(acc_dict[k])
        preds = predict(output_pp,smooth = smooth, filter_size=filter_size)
        if i == 11 or i == 4 or print_all:
            for k in range(batch_size):
                if certainity_map:
                    image_to_mask(input[k,1,:,:].data.cpu().numpy(),\
                              true[k,0,:,:].data.cpu().numpy(),\
                              true[k,1,:,:].data.cpu().numpy(),\
                              true[k,2,:,:].data.cpu().numpy(),\
                             preds[0][k,:,:].cpu().numpy(),\
                             preds[1][k,:,:].cpu().numpy(),\
                             preds[2][k,:,:].cpu().numpy(),\
                            cm[k].data.cpu().numpy())
                else:
                    image_to_mask(input[k,1,:,:].data.cpu().numpy(),\
                              true[k,0,:,:].data.cpu().numpy(),\
                              true[k,1,:,:].data.cpu().numpy(),\
                              true[k,2,:,:].data.cpu().numpy(),\
                             preds[0][k,:,:].cpu().numpy(),\
                             preds[1][k,:,:].cpu().numpy(),\
                             preds[2][k,:,:].cpu().numpy())

    loss = running_loss/data_sizes[phase] 
    dice_score_0 = running_dice_score_class_0/data_sizes[phase]
    dice_score_1 = running_dice_score_class_1/data_sizes[phase]
    dice_score_2 = running_dice_score_class_2/data_sizes[phase]
    for i in range(3):
        dc_sr[i] = list(itertools.chain(*dc_sr[i]))
        acc_sr[i] = list(itertools.chain(*acc_sr[i]))
    print('{} loss: {:.4f}, Dice Score (class 0): {:.4f}, Dice Score (class 1): {:.4f},Dice Score (class 2): {:.4f}'.format(phase,loss, dice_score_0, dice_score_1, dice_score_2))
    return loss, dice_score_0, dice_score_1, dice_score_2, dc_sr, acc_sr