from __future__ import print_function, division
import torch, os, glob
import importlib
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "Utils." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.split('_')[-1] + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls
            
    if dataset is None:
        raise ValueError("In %s.py, there should be a class with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))
    return dataset


def create_dataloader(dataset_name, dataset_dir, batch_size, need_shuffle, is_mpdist=False, world_size=1, rank=0):
    dataset = find_dataset_using_name(dataset_name)
    instance = dataset(dataset_dir)
    if (rank == 0):
        print("[%s] of size %d was created" % (type(instance).__name__, len(instance)))

    if is_mpdist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(instance, num_replicas=world_size, rank=rank, shuffle=need_shuffle)
        dataloader = torch.utils.data.DataLoader(instance,
                                                 batch_size=batch_size,
                                                 sampler=train_sampler,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 drop_last=True)
    else:
        dataloader = torch.utils.data.DataLoader(instance,
                                                 batch_size=batch_size,
                                                 shuffle=need_shuffle,
                                                 num_workers=4,
                                                 drop_last=False)
    return dataloader