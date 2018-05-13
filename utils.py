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

from dataloader_2d import *
from dataloader_3d import *

train_path = '/beegfs/ark576/new_knee_data/train'
val_path = '/beegfs/ark576/new_knee_data/val'
test_path = '/beegfs/ark576/new_knee_data/test'

train_file_names = sorted(pickle.load(open(train_path + '/train_file_names.p','rb')))
val_file_names = sorted(pickle.load(open(val_path + '/val_file_names.p','rb')))
test_file_names = sorted(pickle.load(open(test_path + '/test_file_names.p','rb')))

transformed_dataset = {'train': KneeMRIDataset(train_path,train_file_names, train_data= True, flipping=False, normalize= True),
                       'validate': KneeMRIDataset(val_path,val_file_names, normalize= True),
                       'test': KneeMRIDataset(test_path,test_file_names, normalize= True)
                                          }

dataloader = {x: DataLoader(transformed_dataset[x], batch_size=5,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}

def plot_hist(hist_dict,hist_type,chart_type = 'semi-log'):
    if chart_type == 'log-log':
        plt.loglog(range(len(hist_dict['train'])),hist_dict['train'], label='Train ' + hist_type)
        plt.loglog(range(len(hist_dict['validate'])),hist_dict['validate'], label = 'Validation ' + hist_type)
    if chart_type == 'semi-log':
        plt.semilogy(range(len(hist_dict['train'])),hist_dict['train'], label='Train ' + hist_type)
        plt.semilogy(range(len(hist_dict['validate'])),hist_dict['validate'], label = 'Validation ' + hist_type)
    plt.xlabel('Epochs')
    plt.ylabel(hist_type)
    plt.legend()
    plt.show()

def dice_loss(true,scores, epsilon = 1e-4,p = 2):
    preds = F.softmax(scores)
    N, C, sh1, sh2 = true.size()
    true = true.view(N, C, -1)
    preds = preds.view(N, C, -1)
    wts = torch.sum(true, dim = 2) + epsilon
    wts = 1/torch.pow(wts,p)
    wts = torch.clamp(wts,0,0.1)
    wts[wts == 0.1] = 0
    wts = wts/(torch.sum(wts,dim = 1)[:,None])
    prod = torch.sum(true*preds,dim = 2)
    sum_tnp = torch.sum(true + preds, dim = 2)
    num = torch.sum(wts * prod, dim = 1)
    denom = torch.sum(wts * sum_tnp, dim = 1) + epsilon
    loss = 1 - 2*(num/denom)
    return torch.mean(loss)

def dice_loss_2(true,scores, epsilon = 1e-4,p = 2):
    preds = F.softmax(scores)
    N, C, sh1, sh2 = true.size()
    true = true.view(N, C, -1)
    preds = preds.view(N, C, -1)
    wts = torch.sum(true, dim = 2) + epsilon
    wts = 1/torch.pow(wts,p)
    wts = torch.clamp(wts,0,0.1)
    wts[wts == 0.1] = 1e-6
    wts[:,-1] = 1e-15
    wts = wts/(torch.sum(wts,dim = 1)[:,None])
    prod = torch.sum(true*preds,dim = 2)
    sum_tnp = torch.sum(true + preds, dim = 2)
    num = torch.sum(wts * prod, dim = 1)
    denom = torch.sum(wts * sum_tnp, dim = 1) + epsilon
    loss = 1 - 2*(num/denom)
    return torch.mean(loss)

def segments(seg_1, seg_2, seg_3):
    seg_tot = seg_1 + seg_2 + seg_3
    seg_none = (seg_tot == 0).type(torch.FloatTensor)
    seg_all = torch.cat((seg_1.unsqueeze(1),seg_2.unsqueeze(1),seg_3.unsqueeze(1),seg_none.unsqueeze(1)), dim = 1)
    return seg_all

seg_sum = torch.zeros(3)
for i, data in enumerate(dataloader['train']):
    input, segF, segP, segT,_ = data
    seg_sum[0] += torch.sum(segF)
    seg_sum[1] += torch.sum(segP)
    seg_sum[2] += torch.sum(segT)
mean_s_sum = seg_sum/i

def dice_loss_3(true,scores, epsilon = 1e-4,p = 2, mean=mean_s_sum):
    preds = F.softmax(scores)
    N, C, sh1, sh2 = true.size()
    true = true.view(N, C, -1)
    preds = preds.view(N, C, -1)
    wts = torch.sum(true, dim = 2) + epsilon
    mean = 1/torch.pow(mean,p)
    wts[:,:-1] = mean[None].repeat(N,1)
    wts[:,-1] = 0
    wts = wts/(torch.sum(wts,dim = 1)[:,None])
    prod = torch.sum(true*preds,dim = 2)
    sum_tnp = torch.sum(true + preds, dim = 2)
    num = torch.sum(wts * prod, dim = 1)
    denom = torch.sum(wts * sum_tnp, dim = 1) + epsilon
    loss = 1 - 2*(num/denom)
    return torch.mean(loss)

def predict(scores,smooth = False,filter_size = 3):
    preds = F.softmax(scores)
    pred_class = (torch.max(preds, dim = 1)[1])
    class_0_pred_seg = (pred_class == 0).type(torch.cuda.FloatTensor)
    class_1_pred_seg = (pred_class == 1).type(torch.cuda.FloatTensor)
    class_2_pred_seg = (pred_class == 2).type(torch.cuda.FloatTensor)
    if smooth:
        class_0_pred_seg = F.avg_pool2d(class_0_pred_seg,filter_size,1,int((filter_size-1)/2))>0.5
        class_1_pred_seg = F.avg_pool2d(class_1_pred_seg,filter_size,1,int((filter_size-1)/2))>0.5
        class_2_pred_seg = F.avg_pool2d(class_2_pred_seg,filter_size,1,int((filter_size-1)/2))>0.5
    return class_0_pred_seg.data.type(torch.cuda.FloatTensor), class_1_pred_seg.data.type(torch.cuda.FloatTensor)\
, class_2_pred_seg.data.type(torch.cuda.FloatTensor)


def dice_score(true,scores,smooth = False,filter_size = 3, epsilon = 1e-7):
    N ,C, sh1, sh2 = true.size()
    true = true.view(N,C,-1)
    class_0_pred_seg,class_1_pred_seg,class_2_pred_seg = predict(scores, smooth = smooth,filter_size = filter_size)
    class_0_pred_seg = class_0_pred_seg.view(N,-1)
    class_1_pred_seg = class_1_pred_seg.view(N,-1)
    class_2_pred_seg = class_2_pred_seg.view(N,-1)
    true = true.data.type(torch.cuda.FloatTensor)
    def numerator(truth,pred, idx):
        return(torch.sum(truth[:,idx,:] * pred,dim = 1)) + epsilon/2
    def denominator(truth,pred,idx):
        return(torch.sum(truth[:,idx,:]+pred,dim = 1)) + epsilon
    
    dice_score_class_0 = torch.mean(2*(numerator(true,class_0_pred_seg,0))/(denominator(true,class_0_pred_seg,0)))
    dice_score_class_1 = torch.mean(2*(numerator(true,class_1_pred_seg,1))/(denominator(true,class_1_pred_seg,1)))
    dice_score_class_2 = torch.mean(2*(numerator(true,class_2_pred_seg,2))/(denominator(true,class_2_pred_seg,2)))
    
    return (dice_score_class_0,dice_score_class_1, dice_score_class_2)

def entropy_loss(true,scores,mean = mean_s_sum,epsilon = 1e-4, p=2):
    N,C,sh1,sh2 = true.size()
    wts = Variable(torch.zeros(4).cuda()) + epsilon
    mean = 1/torch.pow(mean,p)
    wts[:-1] = mean
    wts[-1] = 1e-9
    wts = wts/(torch.sum(wts))
    log_prob = F.log_softmax(scores)
    prod = (log_prob*true).view(N,C,-1)
    prod_t = torch.transpose(prod,1,2)
    loss = -torch.mean(prod_t*wts)
    return loss

def image_to_mask(img, femur, patellar, tibia,femur_pr,patellar_pr,tibia_pr,cm = None):
    masked_1 = np.ma.masked_where(femur == 0, femur)
    masked_2 = np.ma.masked_where(patellar == 0,patellar)
    masked_3 = np.ma.masked_where(tibia == 0, tibia)
    
    masked_1_pr = np.ma.masked_where(femur_pr == 0, femur_pr)
    masked_2_pr = np.ma.masked_where(patellar_pr == 0,patellar_pr)
    masked_3_pr = np.ma.masked_where(tibia_pr == 0, tibia_pr)
    masked_cm = np.ma.masked_where(cm ==-1000,cm)
    x = 3
    plt.figure(figsize=(20,10))
    plt.subplot(1,x,1)
    plt.imshow(img, 'gray', interpolation='none')
    plt.subplot(1,x,2)
    plt.imshow(img, 'gray', interpolation='none')
    if np.sum(femur) != 0:
        plt.imshow(masked_1, 'spring', interpolation='none', alpha=0.9)
    if np.sum(patellar) != 0:
        plt.imshow(masked_2, 'coolwarm_r', interpolation='none', alpha=0.9)
    if np.sum(tibia) != 0:
        plt.imshow(masked_3, 'Wistia', interpolation='none', alpha=0.9)
    plt.subplot(1,x,3)
    plt.imshow(img, 'gray', interpolation='none')
    if np.sum(femur_pr) != 0:
        plt.imshow(masked_1_pr, 'spring', interpolation='none', alpha=0.9)
    if np.sum(patellar_pr) != 0:
        plt.imshow(masked_2_pr, 'coolwarm_r', interpolation='none', alpha=0.9)
    if np.sum(tibia_pr) != 0:
        plt.imshow(masked_3_pr, 'Wistia', interpolation='none', alpha=0.9)
    plt.show()
    if cm is not None:
        plt.figure(figsize=(20,20))
        plt.imshow(masked_cm,'coolwarm_r')
        plt.colorbar()
        plt.show()

def generate_noise(true):
    return Variable((2*torch.rand(true.size())-1)*0.1).cuda()

def save_segmentations_2d(model,prediction_models, dataloader,data_sizes,batch_size,phase,model_name,\
                          num_samples = 7, smooth = False, filter_size = 3):
    y_preds = []
    name_list = []
    num_samples = num_samples
    if phase == 'train':
        path = '/beegfs/ark576/Knee Cartilage Data/Train Data/'
    if phase == 'validate':
        path = '/beegfs/ark576/Knee Cartilage Data/Validation Data/'
    if phase == 'test':
        path = '/beegfs/ark576/Knee Cartilage Data/Test Data/'
    for i in prediction_models:
        for param in i.parameters():
            param.requires_grad = False
    
    for i,data in enumerate(dataloader[phase]):
        input, segF,segP, segT,variable_name = data
        input = Variable(input).cuda()
        input_pp = []
        for j in prediction_models:
            output = j(input)
            preds_m = predict_pp(output)
            input_pp.append(preds_m)
        input_pp = torch.cat(input_pp,dim = 1)
        output_pp = model(input_pp)
        preds = predict(output_pp,smooth = smooth, filter_size=filter_size)
        preds = torch.cat((preds[0][:,None],preds[1][:,None],preds[2][:,None]),dim = 1)
        y_preds.append(preds.cpu().numpy())
        name_list.append(variable_name)
    list_of_names = list(itertools.chain(*name_list))
    y_preds = np.concatenate(y_preds).astype(np.uint8)
    for i in range(num_samples):
        name = list_of_names[i*15][:-3]
        pred_segment = y_preds[i*15:(i+1)*15]
        file_name = path + name
        variable = sio.loadmat(file_name)
        temp_variable = {}
        temp_variable['MDnr'] = variable['MDnr']
        preds_all = pred_segment[[0,1,7,8,9,10,11,12,13,14,2,3,4,5,6],:]
        temp_variable['Predicted_segment_F'] = np.transpose(preds_all[:,0,:,:],(1,2,0))
        temp_variable['Predicted_segment_P'] = np.transpose(preds_all[:,1,:,:],(1,2,0))
        temp_variable['Predicted_segment_T'] = np.transpose(preds_all[:,2,:,:],(1,2,0))
        save_path = '/beegfs/ark576/knee-segments/predictions/'+ model_name +'/'+phase+'/'
        sio.savemat(save_path+name,temp_variable,appendmat=False, do_compression=True)

def save_segmentations_2d_prob(model,prediction_models, dataloader,data_sizes,batch_size,phase,model_name,\
                          num_samples = 7, smooth = False, filter_size = 3):
    y_preds = []
    name_list = []
    num_samples = num_samples
    if phase == 'train':
        path = '/beegfs/ark576/Knee Cartilage Data/Train Data/'
    if phase == 'validate':
        path = '/beegfs/ark576/Knee Cartilage Data/Validation Data/'
    if phase == 'test':
        path = '/beegfs/ark576/Knee Cartilage Data/Test Data/'
    for i in prediction_models:
        for param in i.parameters():
            param.requires_grad = False
    
    for i,data in enumerate(dataloader[phase]):
        input, segF,segP, segT,variable_name = data
        input = Variable(input).cuda()
        input_pp = []
        for j in prediction_models:
            output = j(input)
            preds_m = predict_pp(output)
            input_pp.append(preds_m)
        input_pp = torch.cat(input_pp,dim = 1)
        output_pp = model(input_pp)
        preds = F.softmax(output_pp)
        y_preds.append(preds.data.cpu().numpy())
        name_list.append(variable_name)
    list_of_names = list(itertools.chain(*name_list))
    y_preds = np.concatenate(y_preds)
    for i in range(num_samples):
        name = list_of_names[i*15][:-3]
        pred_segment = y_preds[i*15:(i+1)*15]
        file_name = path + name
        variable = sio.loadmat(file_name)
        temp_variable = {}
        temp_variable['NUFnr'] = variable['NUFnr']
        temp_variable['GT_F'] = variable['SegmentationF']
        temp_variable['GT_P'] = variable['SegmentationP']
        temp_variable['GT_T'] = variable['SegmentationT']
        preds_all = pred_segment[[0,1,7,8,9,10,11,12,13,14,2,3,4,5,6],:]
        temp_variable['Predicted_prob'] = preds_all
        save_path = '/beegfs/ark576/knee-segments/predictions/'+ model_name +'/'+phase+'/'
        sio.savemat(save_path+name+'_prob',temp_variable,appendmat=False, do_compression=True)

from sklearn.metrics import confusion_matrix
def dice_score_image(pred,true,epsilon = 1e-5):
    num = 2*np.sum(pred*true) + epsilon
    pred_norm = np.sum(pred)
    true_norm = np.sum(true)
    if pred_norm == 0 or true_norm == 0:
        return None
    else:
        denom = pred_norm + true_norm + epsilon
        return num/denom

def save_segmentations_3D(model, dataloader,data_sizes,batch_size,phase,model_name, num_samples = 7):
    y_preds = []
    name_list = []
    num_samples = num_samples
    if phase == 'train':
        path = '/beegfs/ark576/Knee Cartilage Data/Train Data/'
    if phase == 'validate':
        path = '/beegfs/ark576/Knee Cartilage Data/Validation Data/'
    if phase == 'test':
        path = '/beegfs/ark576/Knee Cartilage Data/Test Data/'
    for data in dataloader[phase]:
        input, segments, variable_name = data
        input = Variable(input).cuda()
        output = model(input)
        output_reshaped = torch.transpose(output,2,1).contiguous().view(-1,4,256,256)
        preds = predict(output_reshaped)
        preds = torch.cat((preds[0][:,None],preds[1][:,None],preds[2][:,None]),dim = 1)
        y_preds.append(preds.cpu().numpy())
        name_list.append(variable_name)
    list_of_names = list(itertools.chain(*name_list))
    y_preds = np.concatenate(y_preds).astype(np.uint8)
    for i in range(num_samples):
        name = list_of_names[i]
        preds_all = y_preds[i*15:(i+1)*15]
        file_name = path + name
        variable = sio.loadmat(file_name)
        temp_variable = {}
        temp_variable['MDnr'] = variable['MDnr']
        temp_variable['Predicted_segment_F'] = np.transpose(preds_all[:,0,:,:],(1,2,0))
        temp_variable['Predicted_segment_T'] = np.transpose(preds_all[:,1,:,:],(1,2,0))
        temp_variable['Predicted_segment_P'] = np.transpose(preds_all[:,2,:,:],(1,2,0))
        save_path = '/beegfs/ark576/knee-segments/predictions/'+ model_name +'/'+phase+'/'
        sio.savemat(save_path+name,temp_variable,appendmat=False, do_compression=True)

def make_certainity_maps(scores):
    probs = F.softmax(scores)
    pred_prob,idx = (torch.max(probs, dim = 1))
    pred_prob_c = torch.clamp(pred_prob,0.00001,0.999999)
    ret_value = torch.log(pred_prob_c) - torch.log((1-pred_prob_c))
    ret_value[idx==3]=-1000
    return ret_value


