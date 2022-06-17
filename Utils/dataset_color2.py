from __future__ import print_function, division
import torch, os, glob
from torch.utils.data import Dataset, DataLoader
from Models import basic
import numpy as np
from PIL import Image
import cv2


class Color2Dataset(Dataset):

    def __init__(self, root_dir, loop_round=1):
        """
        Args:
            root_dir (string): directory consisting of three image folders
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print('Warning@@@: dataset',root_dir, 'NOT exist.----------')
            return
        self.file_list = glob.glob(os.path.join(root_dir, '*.*'))
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rgb_img = np.array(Image.open(self.file_list[idx]).convert("RGB"), np.float32) / 255.
        rgb_img = torch.from_numpy(rgb_img.transpose((2, 0, 1)))
        #! RGB (0.0~1.0) to Lab (-1.0~1.0)
        rgb_img = rgb_img.unsqueeze(dim=0)
        lab_img = basic.rgb2lab(rgb_img)
        gray_img = lab_img[0,0:1,:,:]
        color_map = lab_img[0,1:3,:,:]
        return {'grays': gray_img, 'ABs': color_map, 'colors': rgb_img[0,:,:,:]*2.-1.}