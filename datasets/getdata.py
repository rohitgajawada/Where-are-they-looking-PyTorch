from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.patches as patches
import scipy.io as sio

def getCropped(img, e):

    alpha = 0.3
    w_x = int(math.floor(alpha * img.shape[1]))
    w_y = int(math.floor(alpha * img.shape[0]))

    if w_x % 2 == 0:
        w_x = w_x + 1

    if w_y % 2 == 0:
        w_y = w_y + 1

    im_face = np.ones((w_y, w_x, 3))
    im_face[:, :, 0] = 123 * np.ones((w_y, w_x))
    im_face[:, :, 1] = 117 * np.ones((w_y, w_x))
    im_face[:, :, 2] = 104 * np.ones((w_y, w_x))

    center = [math.floor(e[0] * img.shape[1]), math.floor(e[1] * img.shape[0])]
    d_x = math.floor((w_x - 1) / 2)
    d_y = math.floor((w_y - 1) / 2)

    bottom_x = center[0] - d_x - 1
    delta_b_x = 0
    if bottom_x < 0:
        delta_b_x = 1 - bottom_x
        bottom_x = 0

    top_x = center[0] + d_x - 1
    delta_t_x = w_x - 1
    if top_x > img.shape[1] - 1:
        delta_t_x = w_x - (top_x - img.shape[1] + 1)
        top_x = img.shape[1] - 1

    bottom_y = center[1] - d_y - 1
    delta_b_y = 0
    if bottom_y < 0:
        delta_b_y = 1 - bottom_y
        bottom_y = 0

    top_y = center[1] + d_y - 1
    delta_t_y = w_y - 1
    if top_y > img.shape[0] - 1:
        delta_t_y = w_y - (top_y - img.shape[0] + 1)
        top_y = img.shape[0] - 1


    x = img[int(bottom_y): int(top_y + 1), int(bottom_x): int(top_x + 1), :]
    x = np.ascontiguousarray(x)

    try:
        x = transform.resize(x,(227, 227))
        return x
    except:
        print('begins')
        print(x)
        print(x.shape)
        print('ends')
        return img


class GazeDataset(Dataset):
    ##THERE IS NO FRIKKING DATA AUG IN THIS OLD CODE!!! slap the old me from 2 yrs pls

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

        bbox = np.ascontiguousarray(bbox)
        bbox = transform.resize(bbox,(227, 227))

        img = np.ascontiguousarray(img)
        img = transform.resize(img,(227, 227))

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img + self.places_mean)
        # plt.show()
        # exit()

        gaze = self.gaze_list[idx]
        eyes = self.eyes_list[idx]

        eyes2 = (eyes - bbox_corr[:2])/bbox_corr[2:]

        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)

        if len(bbox.shape) == 2:
            bbox = np.stack((bbox,) * 3, axis=-1)

        bbox = getCropped(bbox, eyes2)
        bbox = np.ascontiguousarray(bbox)
        bbox = transform.resize(bbox,(227, 227))

        eyes_loc_size = 13
        gaze_label_size = 5

        eyes_loc = np.zeros((eyes_loc_size, eyes_loc_size))
        eyes_loc[int(np.floor(eyes_loc_size * eyes[1]))][int(np.floor(eyes_loc_size * eyes[0]))] = 1

        gaze_label = np.zeros((gaze_label_size,gaze_label_size))

        gaze_label[int(np.floor(gaze_label_size * gaze[1]))][int(np.floor(gaze_label_size * gaze[0]))] = 1

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous()

        bbox = bbox.transpose((2, 0, 1))
        bbox = torch.from_numpy(bbox).contiguous()

        normtransform = transforms.Compose([
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

        img = normtransform(img)
        bbox = normtransform(bbox)

        eyes_loc = torch.from_numpy(eyes_loc).contiguous()
        gaze_label = torch.from_numpy(gaze_label).contiguous()
        
        gaze_label = gaze_label.view(1, 25)
        # print(gaze.shape)
        gaze_final = np.ones(100)
        gaze_final *= -1
        gaze_final[:gaze.shape[0]] = gaze
        sample = (img.float(), bbox.float(), eyes_loc.float(), gaze_label.float(), eyes, idx, eyes2, gaze_final)

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
