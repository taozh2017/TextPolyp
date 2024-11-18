#!/usr/bin/python3
# coding=utf-8

import os
import sys

# sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from skimage import img_as_ubyte
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import dataset
import time
import logging as logger
from Eval.eval_functions import *
from lib.Net import Net

TAG = "res2net"
SAVE_PATH = TAG
GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', \
                   filename="test_%s.log" % (TAG), filemode="w")

# DATASETS = ['./data/ECSSD', './data/DUT-OMRON', './data/PASCAL-S', './data/HKU-IS', './data/THUR15K', './data/DUTS', ]
DATASETS = ['./data/TestDataset/CVC-ClinicDB',
            './data/TestDataset/CVC-300',
            './data/TestDataset/CVC-ColonDB',
            './data/TestDataset/Kvasir',
            './data/TestDataset/ETIS-LaribPolypDB']


def Dice(pred, mask):
    inter = (pred * mask).sum()
    union = (pred + mask).sum()
    dice1 = (2 * inter + 1e-10) / (union + 1e-10)
    return dice1


class Test(object):
    def __init__(self, Dataset, datapath, Network):
        ## dataset
        self.datapath = datapath.split("/")[-1]

        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=8)
        ## network
        # self.net = Network(self.cfg)
        self.net = Net()
        path = './Models/Final/' + 'best.pt'
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)

        self.net.train(False)
        self.net.cuda()

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number, dice, Sm = 0, 0, 0, 256, 0, 0
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, mask, (H, W), maskpath in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                mask = torch.where(mask > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
                start_time = time.time()
                pred = self.net(image, 'test')
                # pred = torch.sigmoid(pred)
                torch.cuda.synchronize()
                end_time = time.time()
                cost_time += end_time - start_time
                ## MAE
                cnt += 1
                ##Dice
                dice += Dice(pred, mask)
                mae += (pred - mask).abs().mean()
                # sm
                Sm += StructureMeasure(pred, mask)
                ## F-Score
                precision = torch.zeros(number)
                recall = torch.zeros(number)
                for i in range(number):
                    temp = (pred >= threshod[i]).float()
                    precision[i] = (temp * mask).sum() / (temp.sum() + 1e-12)
                    recall[i] = (temp * mask).sum() / (mask.sum() + 1e-12)
                mean_pr += precision
                mean_re += recall
                fscore = mean_pr * mean_re * (1 + 0.3) / (0.3 * mean_pr + mean_re + 1e-12)
                if cnt % 10 == 0:
                    fps = image.shape[0] / (end_time - start_time)
                    print('Dice =%.6f, MAE=%.6f, Sm=%.6f, fps=%.4f' % (
                     dice / cnt, mae / cnt, Sm / cnt, fps))
            fps = len(self.loader.dataset) / cost_time
            msg = ' %s, Dice =%.6f,MAE=%.6f, Sm=%.6f, len(imgs)=%s, fps=%.4f' % (
                 self.datapath, dice / cnt, mae / cnt, Sm / cnt, len(self.loader.dataset), fps)
            # print(msg)
            logger.info(msg)


    def save(self):
        with torch.no_grad():
            for image, mask, (H, W), name in self.loader:
                out2 = self.net(image.cuda().float(), 'test')
                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                pred = (out2[0, 0]).cpu().numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                head = './map/eval/{}/'.format(TAG) + self.cfg.datapath.split('/')[-1]
                # head = './map/Eval/{}/{}/'.format(TAG, s) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))


if __name__ == '__main__':
    for d in DATASETS:
        t = Test(dataset, d, Net)
        t.accuracy()
        t.save()
