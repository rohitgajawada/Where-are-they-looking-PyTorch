import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class RandomHorizontalFlip(object):

    def __init__(self):
        pass

    def __call__(self, sample):

        image = sample['img']
        bbox = sample['bbox']
        eyes = sample['eyes']
        eyes_bbox = sample['eyes_bbox']
        gaze = sample['gaze']

        h, w = image.shape[:2]
        h_bbox, w_bbox = bbox.shape[:2]

        if random.random() > 0.5:

            image = image[:, ::-1]
            bbox = bbox[:, ::-1]
            eyes[1] = 1 - eyes[1]
            eyes_bbox[1] = 1 - eyes_bbox[1]
            gaze[1] = 1 - gaze[1]

        sample = {'img': image, 'bbox': bbox, 'eyes': eyes, 'eyes_bbox': eyes_bbox, 'gaze': gaze}

        return sample


class RandomCrop(object):

    def __init__(self, crop_size, orig_size):

        self.crop_size = (crop_size, crop_size)
        self.orig_size = (orig_size, orig_size)

    def __call__(self, sample):

        image = sample['img']
        bbox = sample['bbox']
        eyes = sample['eyes']
        eyes_bbox = sample['eyes_bbox']
        gaze = sample['gaze']

        if random.random() > 0.05:

            h, w = image.shape[:2]
            new_h, new_w = self.crop_size
            orig_h, orig_w = self.orig_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h, left: left + new_w]
            image = transform.resize(image, (orig_h, orig_w))

            eyes = (eyes * orig_w - [top, left]) / (1.0 * orig_w)   ##assuming that orig_w is equal to orig_h
            eyes_bbox = (eyes_bbox * orig_w - [top, left]) / (1.0 * orig_w)
            gaze = (gaze * orig_w - [top, left]) / (1.0 * orig_w)

        sample = {'img': image, 'bbox': bbox, 'eyes': eyes, 'eyes_bbox': eyes_bbox, 'gaze': gaze}

        return sample
