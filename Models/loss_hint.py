from __future__ import division
import os, glob, shutil, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as VM
from Models import basic
from Utils import util


class HintLoss:
    def __init__(self, mpdist=False, gpu_no=0):
        self.mpdist = mpdist
        self.gpu_no = gpu_no
    
    def get_loss(self, data, epoch_no):
        sparseLoss_idx = data['sparse_loss'].mean()
        #valueLoss_idx = binaryValueLoss(data['hint_masks'])
        ReconLoss_idx = l1_loss(data['pred_ABs'], data['target_ABs'])
        return {'sparseLoss':sparseLoss_idx, 'reconLoss':ReconLoss_idx}


def l2_loss(y_input, y_target):
    return F.mse_loss(y_input, y_target)


def l1_loss(y_input, y_target):
    return F.l1_loss(y_input, y_target)


def binaryValueLoss(yInput):
    # data range is [-1,1]
    return (yInput.abs() - 1.0).abs().mean()


def huber_loss(y_input, y_target, delta=0.01):
    mask = torch.zeros_like(y_input)
    mann = torch.abs(y_input - y_target)
    eucl = 0.5 * (mann**2)
    mask[...] = mann < delta
    loss = eucl * mask / delta + (mann - 0.5 * delta) * (1 - mask)
    return torch.mean(loss)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()