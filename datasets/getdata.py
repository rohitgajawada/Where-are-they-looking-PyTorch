from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.patches as patches
import scipy.io as sio

class GazeDataset(Dataset):

    def __init__(self, Data, transform=None):

        data_bbox = Data['train_bbox'][0]
        data_eyes = Data['train_eyes'][0]
        data_gaze = Data['train_gaze'][0]
        data_meta = Data['train_meta'][0]
        data_path = Data['train_path']

        for i in range(data_path.shape[0]):
            data_path[i] = data_path[i][0]
        data_path = data_path.flatten()
        data_path = 'data/' + data_path

        for i in range(data_bbox.shape[0]):
            data_bbox[i] = data_bbox[i].flatten()

        for i in range(data_eyes.shape[0]):
            data_eyes[i] = data_eyes[i].flatten()

        for i in range(data_gaze.shape[0]):
            data_gaze[i] = data_gaze[i].flatten()

        self.img_path_list = data_path
        self.bbox_list = data_bbox
        self.eyes_list = data_eyes
        self.gaze_list = data_gaze
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img = io.imread(img_name)
        gaze = self.gaze_list[idx]
        bbox = self.bbox_list[idx]
        eyes = self.eyes_list[idx]
        sample = {'image':img, 'gaze':gaze, 'bbox':bbox, 'eyes':eyes}

        if self.transform:
            sample = self.transform(sample)

        return sample

Train_Ann = sio.loadmat('data/train_annotations.mat')
Test_Ann = sio.loadmat('data/test_annotations.mat')

train_gaze = GazeDataset(Train_Ann)

for i in range(len(train_gaze)):
    if i <= 10:
        sample = train_gaze[i]
        print(i, sample['image'].shape)
        I = sample['image']
        s = sample['image'].shape
        bbox = sample['bbox']
        print(bbox)
        eyes = sample['eyes']
        gaze = sample['gaze']
        fig,ax = plt.subplots(1)
        ax.imshow(I)

        rect = patches.Rectangle((bbox[0]*s[1], bbox[1]*s[0]), bbox[2]*s[1], bbox[3]*s[0],linewidth=1,edgecolor='r',facecolor='none')

        # rect = patches.Rectangle((eyes[0]*s[1], eyes[1]*s[0]),5,5 ,linewidth=1,edgecolor='r',facecolor='none')

        ax.plot((eyes[0]*s[1], gaze[0]*s[1]),(eyes[1]*s[0], gaze[1]*s[0]))
        ax.add_patch(rect)
        plt.show()
    else:
        break
