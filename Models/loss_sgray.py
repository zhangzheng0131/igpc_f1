import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as VM
from Models import basic
from Models.model import *
import os, glob, shutil, math, json
from math import exp

eps = 0.0000001

class SGrayLoss:
    def __init__(self, mpdist=False, gpu_no=0):
        self.mpdist = mpdist
        self.gpu_no = gpu_no
        self.vgg_loss = VGG19Loss(mpdist, gpu_no)
    
    def get_loss(self, data, epoch_no):
        #! invertibility loss
        reconLoss_idx = l2_loss(data['pred_color'], data['target_color'])
        #! gradient loss
        ref_gray = basic.rgb2gray(data['target_color'])
        #tvar_pred = computeTotalVariation(data['pred_gray'])
        #tvar_refe = computeTotalVariation(ref_gray)
        gradLoss_idx = computeLocalVariation(data['pred_gray'], ref_gray)
        #! brightness mapping loss
        '''
        diff_map = (data['pred_gray']-ref_gray).abs()
        mask = diff_map.gt(70/127.0).float()
        norm_divider = mask.sum()+eps
        mappingLoss_idx = (diff_map*mask).sum()/norm_divider - (mask.sum()/norm_divider)*70/127.0
        '''
        thresMap = torch.tensor([70/127.0]).cuda(ref_gray.get_device())
        mappingLoss_idx = torch.max(thresMap, (data['pred_gray']-ref_gray).abs()).mean() - 70/127.0
        #! contrast loss
        N,C,H,W = data['pred_gray'].shape
        pred_grayRGB = data['pred_gray'].expand(N,3,H,W)
        contrastLoss_idx = self.vgg_loss(pred_grayRGB*0.5+0.5, data['target_color']*0.5+0.5)
        #print('loss terms:', reconLoss_idx.item(), gradLoss_idx.item(), mappingLoss_idx.item(), contrastLoss_idx.item())
        return {'reconLoss':reconLoss_idx, 'gradLoss':gradLoss_idx, 'mappingLoss':mappingLoss_idx,\
         'contrastLoss':contrastLoss_idx}


def l2_loss(y_input, y_target):
    return F.mse_loss(y_input, y_target)


def l1_loss(y_input, y_target):
    return F.l1_loss(y_input, y_target)
    

def binaryValueLoss(yInput):
    # data range is [-1,1]
    return (yInput.abs() - 1.0).abs().mean()


def computeTotalVariation(gray_batch):
    """
    Implemented as the variant of TensorFlow API: tf.image.total_variation
    """
    y_tv = ((gray_batch[:,:,1:,:]-gray_batch[:,:,:-1,:]).abs()).mean()
    x_tv = ((gray_batch[:,:,:,1:]-gray_batch[:,:,:,:-1]).abs()).mean()
    totalVar = x_tv + y_tv
    return totalVar

def computeLocalVariation(gray_batch1, gray_batch2):
    """
    Implemented as the variant of TensorFlow API: tf.image.total_variation
    """
    y_tv1 = (gray_batch1[:,:,1:,:]-gray_batch1[:,:,:-1,:])
    x_tv1 = (gray_batch1[:,:,:,1:]-gray_batch1[:,:,:,:-1])
    y_tv2 = (gray_batch2[:,:,1:,:]-gray_batch2[:,:,:-1,:])
    x_tv2 = (gray_batch2[:,:,:,1:]-gray_batch2[:,:,:,:-1])
    diff_var = (x_tv1-x_tv2).abs().mean() + (y_tv1-y_tv2).abs().mean()
    return diff_var


class VGG19Loss:
    def __init__(self, distribute_mode=False, gpu_no=0):
        # data in RGB format, [0,1] range
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        vgg19 = VM.vgg19(pretrained=True)
        # vgg19 = vgg.vgg19(pretrained=True)
        # feature map after conv4_1 (conv & relu)
        self.featureExactor = nn.Sequential(*list(vgg19.features)[:21])
        for param in self.featureExactor.parameters():
            param.requires_grad = False

        if distribute_mode:
            self.featureExactor = torch.nn.DataParallel(self.featureExactor, device_ids=[gpu_no]).cuda()
        else:
            self.featureExactor = torch.nn.DataParallel(self.featureExactor).cuda()
        #! evaluation mode
        self.featureExactor.eval()
        print('[*] Vgg19Loss init!')

    def normalize(self, tensor):
        tensor = tensor.clone()
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor
        
    def get_feature(self, yInput):
        return self.featureExactor(self.normalize(yInput))

    def __call__(self, yInput, yTarget):
        inFeature = self.featureExactor(self.normalize(yInput))
        targetFeature = self.featureExactor(self.normalize(yTarget))
        return l2_loss(inFeature, targetFeature)
