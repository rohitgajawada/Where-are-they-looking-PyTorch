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

    def __call__(self, image, bbox, eyes, eyes_bbox, gaze):

        h, w = image.shape[:2]
        h_bbox, w_bbox = bbox.shape[:-2]

        if random.random() > 0.5:

            image = image[:, ::-1]
            bbox = bbox[:, ::-1]
            eyes[0] = w - eyes[0]
            eyes_bbox[0] = w - eyes_bbox[0]
            gaze[0] = w - gaze[0]

        return image, bbox, eyes, eyes_bbox, gaze


class RandomCrop(object):

    def __init__(self, crop_size, orig_size):

        self.crop_size = (crop_size, crop_size)
        self.orig_size = (orig_size, orig_size)

    def __call__(self, image, bbox, eyes, eyes_bbox, gaze):

        if random.random() > 0.5:

            h, w = image.shape[:2]
            new_h, new_w = self.crop_size
            orig_h, orig_w = self.orig_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h, left: left + new_w]
            image = transform.resize(image, (orig_h, orig_w))

            eyes = eyes - [left, top]
            eyes_bbox = eyes_bbox - [left, top]
            eyes_bbox = eyes_bbox - [left, top]
            gaze = gaze - [left, top]

        return image, bbox, eyes, eyes_bbox, gaze
