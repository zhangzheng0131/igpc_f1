from __future__ import division
import os, glob, shutil, math, json
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as VM
from Models import basic
from Utils import util

class GLoss:
    def __init__(self, train_dict, save_dir, mpdist=False, gpu_no=0):
        self.loss_dict = train_dict['loss_weight']
        self.loss_list = {}
        self.epoch_loss = {}
        self.niters = 0
        for key in self.loss_dict:
            self.loss_list[key] = []
            self.epoch_loss[key] = 0
        self.save_dir = save_dir
        if gpu_no == 0:
            util.exists_or_mkdir(self.save_dir, need_remove=False)
        loss_func_class = find_loss_func_with_name(train_dict['loss_func'])
        self.loss_func = loss_func_class(mpdist, gpu_no)

    def get_epoch_losses(self):
        epoch_loss = self.epoch_loss['totalLoss'] / self.niters
        #! save epoch loss in the list and reset for next epoch
        for key in self.loss_dict:
            self.loss_list[key].append(self.epoch_loss[key] / self.niters)
            self.epoch_loss[key] = 0
        self.niters = 0
        return epoch_loss
    
    def save_epoch_losses(self, resume_mode=False):
        for key in self.loss_dict:
            util.save_list(os.path.join(self.save_dir, key), self.loss_list[key], resume_mode)

    def __call__(self, data, epoch_no):
        loss_terms = self.loss_func.get_loss(data, epoch_no)
        totalLoss_idx = 0
        #print('-------here:', self.loss_dict['sparseLoss'] )
        for key in loss_terms:
            totalLoss_idx += self.loss_dict[key] * loss_terms[key]
            self.epoch_loss[key] += self.loss_dict[key] * loss_terms[key].item()
        self.epoch_loss['totalLoss'] += totalLoss_idx.item()
        self.niters += 1
        return totalLoss_idx


def l2_loss(y_input, y_target):
    return F.mse_loss(y_input, y_target)


def l1_loss(y_input, y_target):
    return F.l1_loss(y_input, y_target)


def huber_loss(y_input, y_target, delta=0.01):
    mask = torch.zeros_like(y_input)
    mann = torch.abs(y_input - y_target)
    eucl = 0.5 * (mann**2)
    mask[...] = mann < delta
    loss = eucl * mask / delta + (mann - 0.5 * delta) * (1 - mask)
    return torch.mean(loss)


def find_loss_func_with_name(loss_name):
    loss_filename = "Models." + loss_name
    datasetlib = importlib.import_module(loss_filename)
    loss_class = None
    target_loss_name = loss_name.split('_')[-1] + 'loss'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_class = cls
    if loss_class is None:
        raise ValueError("In %s.py, there should be a class with class name that matches %s in lowercase." %
                         (loss_filename, target_loss_name))
    return loss_class