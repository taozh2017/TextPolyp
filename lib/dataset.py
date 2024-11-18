#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np

try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))
        if 'TrainDB1' in self.kwargs['datapath']:
            self.mean = np.array([[[126.57, 76.71, 54.92]]])
            self.std = np.array([[[81.53, 55.24, 43.98]]])
        elif 'CVC-300' in self.kwargs['datapath']:
            self.mean = np.array([[[117.20, 77.57, 61.88]]])
            self.std = np.array([[[78.79, 58.32, 48.90]]])
        elif 'CVC-ClinicDB' in self.kwargs['datapath']:
            self.mean = np.array([[[106.54, 70.57, 47.06]]])
            self.std = np.array([[[76.42, 50.91, 33.18]]])
        elif 'CVC-ColonDB' in self.kwargs['datapath']:
            self.mean = np.array([[[112.41, 73.09, 47.80]]])
            self.std = np.array([[[79.22, 58.37, 42.32]]])
        elif 'ETIS-LaribPolypDB' in self.kwargs['datapath']:
            self.mean = np.array([[[153.37, 109.91, 94.88]]])
            self.std = np.array([[[67.40, 60.72, 56.59]]])
        elif 'Kvasir' in self.kwargs['datapath']:
            self.mean = np.array([[[143.38, 83.18, 61.45]]])
            self.std = np.array([[[81.50, 57.41, 48.67]]])
        else:
            # raise ValueError
            self.mean = np.array([[[0.485 * 256, 0.456 * 256, 0.406 * 256]]])
            self.std = np.array([[[0.229 * 256, 0.224 * 256, 0.225 * 256]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        with open(os.path.join(cfg.datapath, cfg.mode + '.txt'), 'r') as lines:
            self.samples = []
            for line in lines:
                imagepath = os.path.join(cfg.datapath, 'images', line.strip() + '.png')
                maskpath = os.path.join(cfg.datapath, 'masks', line.strip() + '.png')

                self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(320, 320),
                                               transform.RandomHorizontalFlip(),
                                               transform.RandomCrop(288, 288),
                                               transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(320, 320),
                                               transform.ToTensor())
        elif cfg.mode == 'val':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(320, 320),
                                               transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath = self.samples[idx]

        image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
        mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
        # label = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]
        H, W, C = mask.shape
        image, mask = self.transform(image, mask)
        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.ion()

    cfg = Config(mode='train', datapath='./data/TrainDB1')
    data = Data(cfg)
    for i in range(100):
        image, depth, mask = data[i]
        image = image.permute(1, 2, 0).numpy() * cfg.std + cfg.mean
        mask = mask.numpy()
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
