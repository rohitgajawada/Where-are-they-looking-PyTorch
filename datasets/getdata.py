from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import matplotlib.patches as patches
import scipy.io as sio

class GazeDataset(Dataset):

    def __init__(self, Data, type, path):

        if type == 'train':
            data_bbox = Data['train_bbox'][0]
            data_eyes = Data['train_eyes'][0]
            data_gaze = Data['train_gaze'][0]
            data_meta = Data['train_meta'][0]
            data_path = Data['train_path']

        if type == 'test':
            data_bbox = Data['test_bbox'][0]
            data_eyes = Data['test_eyes'][0]
            data_gaze = Data['test_gaze'][0]
            data_meta = Data['test_meta'][0]
            data_path = Data['test_path']

        for i in range(data_path.shape[0]):
            data_path[i] = data_path[i][0]
        data_path = data_path.flatten()
        data_path = path + data_path

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

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img = io.imread(img_name)
        s = img.shape
        bbox_corr = self.bbox_list[idx]
        bbox_corr[bbox_corr < 0] = 0.0
        bbox = np.copy(img[ int(bbox_corr[1] * s[0]): int(np.ceil( bbox_corr[1] * s[0] + bbox_corr[3] * s[0])), int(bbox_corr[0] * s[1]): int(np.ceil(bbox_corr[0] * s[1] + bbox_corr[2] * s[1]))])

        bbox = transform.resize(bbox,(227, 227))

        img = transform.resize(img,(227, 227))
        gaze = self.gaze_list[idx]
        eyes = self.eyes_list[idx]

        eyes_loc_size = 13
        gaze_label_size = 14

        eyes_loc = np.zeros((eyes_loc_size, eyes_loc_size))
        eyes_loc[int(eyes_loc_size * eyes[1])][int(eyes_loc_size * eyes[0])] = 1

        gaze_label = np.zeros((gaze_label_size + 1) * (gaze_label_size + 1))
        gaze_label[int(np.floor(gaze_label_size * gaze_label_size * gaze[0] + gaze_label_size*gaze[1]))] = 1

        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous()

        if len(bbox.shape) == 2:
            bbox = np.stack((bbox,) * 3, axis=-1)

        bbox = bbox.transpose((2, 0, 1))
        bbox = torch.from_numpy(bbox).contiguous()

        eyes_loc = torch.from_numpy(eyes_loc).contiguous()
        gaze_label = torch.from_numpy(gaze_label).contiguous()

        sample = (img.float(), bbox.float(), eyes_loc.float(), gaze_label.float())

        return sample

class GazeFollow():

    def __init__(self, opt):

        Train_Ann = sio.loadmat(opt.data_dir + 'train_annotations.mat')
        Test_Ann = sio.loadmat(opt.data_dir + 'test_annotations.mat')

        self.train_gaze = GazeDataset(Train_Ann, 'train', opt.data_dir)
        self.x = self.train_gaze[1]
        self.train_loader = torch.utils.data.DataLoader(self.train_gaze, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

        self.val_gaze = GazeDataset(Test_Ann, 'test', opt.data_dir)
        self.val_loader = torch.utils.data.DataLoader(self.val_gaze,
        batch_size=opt.testbatchsize, shuffle=True, num_workers=opt.workers)
