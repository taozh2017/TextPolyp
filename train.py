# coding=utf-8
import pdb
import sys
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from data import dataset
from lib import dataset as val_dataset
from lib.Net import Net
from data.data_prefetcher import DataPrefetcher
from data.data_prefetcher import DataPrefetcher_val
from loss import *
import logging as logger
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
from lib.tools import *
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
from typing import Any, Dict, List
from segment_anything.util import get_boxes_from_mask
import time

torch.autograd.set_detect_anomaly(True)
sys.dont_write_bytecode = True
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

TAG = "res2net"
Val_PATH = './data/ValDB/'
SAVE_PATH = './Models/{}/'.format(TAG)
temp_pic = './out_F/{}/net/'.format(TAG)
temp_sam = './out_F/{}/sam/'.format(TAG)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', \
                   filename="train_%s.log" % (TAG), filemode="w")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

""" set lr """


def Dice(pred, mask):
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()
    dice = (2.0 * inter) / (union + 1e-10)
    return dice


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
                    annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps * ratio)
    last = total_steps - first
    min_lr = base_lr * annealing_decay
    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1. - x)
        else:
            momentum = momentums[0]

    return lr, momentum


BASE_LR = 1e-5
MAX_LR = 1e-2
batch = 12
l = 0.3


def train(Dataset, Network, args):
    ## dataset
    cfg = Dataset.Config(datapath='./data/TrainDB', savepath=SAVE_PATH, mode='train', batch=batch, lr=1e-3,
                         momen=0.9,
                         decay=5e-4, epoch=100)
    data = Dataset.Data(cfg)
    train_loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)

    ## network
    net = Network
    print("Loading sam model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    for n, p in sam.prompt_encoder.named_parameters():
        p.requires_grad = False
    for n, p in sam.image_encoder.named_parameters():
        p.requires_grad = False
    for n, p in sam.image_encoder.blocks[10].named_parameters():
        p.requires_grad = True
    for n, p in sam.image_encoder.blocks[11].named_parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')
    sam.cuda()
    sam.train()
    net.train()
    net.cuda()
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone1' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    global_step = 0
    db_size = len(train_loader)
    criterion.cuda()
    best_epoch = -1
    best_dice = 0
    # -------------------------- training ------------------------------------#
    for epoch in range(cfg.epoch):
        net.train()
        sam.train()
        prefetcher = DataPrefetcher(train_loader)
        batch_idx = -1
        image, mask, img_sam, gama, pseudo, size, names = prefetcher.next()
        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch * db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1

            ######   pseudo label   ######
            gt = mask.squeeze(1).long()
            bg_label = gt.clone()
            fg_label = gt.clone()
            sam_label = gt.clone()
            bg_label[gt != 0] = 255
            fg_label[gt == 0] = 255
            sam_label[gt == 255] = 0

            ######  saliency structure consistency loss  #####
            out2 = net(image, 'train')

            loss_point = criterion(out2, fg_label) + criterion(out2, bg_label)
            pred_label = torch.squeeze(out2[:, 1:2], dim=1)
            # pdb.set_trace()
            list_s = []
            list_s2 = []

            for i, pack in enumerate(zip(img_sam, gama, sam_label, pred_label)):
                img, img_g, scribble, pred = pack
                box_s = get_boxes_from_mask(scribble.detach().cpu())
                box_p = get_boxes_from_mask((pred >= 0.8).to(torch.float).detach().cpu())
                box = Union_box(box_s, box_p, 120)
                dict1 = {'image': img, 'original_size': (320, 320), 'boxes': box.cuda()}
                list_s.append(dict1)
                dict2 = {'image': img_g, 'original_size': (320, 320), 'boxes': box.cuda()}
                list_s2.append(dict2)
            out1_sam = sam(list_s, multimask_output=False)
            out2_sam = sam(list_s2, multimask_output=False)
            sam_o = []
            indicator1 = []
            for i in range(len(out1_sam)):
                mask, flag = Filter(out1_sam[i]['masks'], pseudo[i])
                sam_o.append(mask)
                indicator1.append(flag)
            indicator1 = torch.IntTensor(indicator1)
            label_sam1 = torch.cat(sam_o, dim=0)
            sam_o2 = []
            indicator2 = []
            for i in range(len(out2_sam)):
                mask, flag = Filter(out2_sam[i]['masks'], pseudo[i])
                sam_o2.append(mask)
                indicator2.append(flag)
            label_sam2 = torch.cat(sam_o2, dim=0)
            weit1, weit2 = uncertainty(label_sam1, label_sam2)
            indicator2 = torch.IntTensor(indicator2)
            indicator = indicator1 & indicator2
            f = 0
            if torch.all(indicator == 0).item() == False:
                loss_sam1 = SaliencyStructureConsistency(label_sam1[indicator == 1].float(),
                                                         pseudo[indicator == 1].float(), 1)
                loss_sam2 = SaliencyStructureConsistency(label_sam2[indicator == 1].float(),
                                                         pseudo[indicator == 1].float(), 1)
                loss_sam = 0.5 * (loss_sam1 + loss_sam2)

                loss_ssc = SaliencyStructureConsistency(label_sam1.float(), label_sam2.float(), 0.85)

                loss2_sam1 = structure_loss(out2[:, 1:2][indicator == 1], label_sam1[indicator == 1].float(),
                                            weit1[indicator == 1])
                loss2_sam2 = structure_loss(out2[:, 1:2][indicator == 1], label_sam2[indicator == 1].float(),
                                            weit2[indicator == 1])
                loss2_sam = 0.5 * (loss2_sam1 + loss2_sam2)

                loss_pseudo = SaliencyStructureConsistency(out2[:, 1:2][indicator == 1],
                                                           pseudo[indicator == 1].float(), 1)

                loss = loss_point + loss_pseudo + loss2_sam + loss_sam + loss_ssc
                f = 1
                if batch_idx % 20 == 0:
                    msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f |  loss_point=%.6f | loss_sam=%.6f  | loss_ssc=%.6f | loss2_sam=%.6f | loss_pseudo=%.6f | flag=%d' % (
                        SAVE_PATH, datetime.datetime.now(), global_step, epoch + 1, cfg.epoch,
                        optimizer.param_groups[1]['lr'],
                        loss.item(), loss_point.item(), loss_sam.item(), loss_ssc.item(), loss2_sam.item(),
                        loss_pseudo.item(),
                        f)
                    print(msg)
                    logger.info(msg)
            else:
                loss_pseudo = SaliencyStructureConsistency(out2[:, 1:2], pseudo.float(), 1)

            loss = loss_point + loss_pseudo
            if batch_idx % 20 == 0:
                msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f |  loss_point=%.6f | loss_pseudo=%.6f' % (
                    SAVE_PATH, datetime.datetime.now(), global_step, epoch + 1, cfg.epoch,
                    optimizer.param_groups[1]['lr'],
                    loss.item(), loss_point.item(), loss_pseudo.item())
                print(msg)
                logger.info(msg)
                if batch_idx % 20 == 0:
                    msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f |  loss_point =%.6f | loss_sam=%.6f  | loss_ssc=%.6f | loss2_sam=%.6f | loss_pseudo=%.6f | flag=%d' % (
                        SAVE_PATH, datetime.datetime.now(), global_step, epoch + 1, cfg.epoch,
                        optimizer.param_groups[1]['lr'],
                        loss.item(), loss.item(), loss.item(), loss.item(), loss.item(), loss.item(),
                        f)
                    print(msg)
                    logger.info(msg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            image, mask, img_sam, gama, pseudo, size, names = prefetcher.next()

        # if (epoch + 1) % 1 == 0 or (epoch + 1) == cfg.epoch:
        #     latest_dice = val(val_dataset, net)
        #     msg_ = 'latest-epoch: %d | dice: %.6f' % ((epoch + 1), latest_dice)
        #     logger.info(msg_)
        #     if not os.path.exists(cfg.savepath):
        #         os.makedirs(cfg.savepath)
        #     if latest_dice > best_dice:
        #         best_dice = latest_dice
        #         best_epoch = epoch + 1
        #         torch.save(net.state_dict(), cfg.savepath + 'best.pt')
        #         msg_best = 'best-epoch: %d |best-dice: %.6f' % (best_epoch, best_dice)
        #         logger.info(msg_best)
        if not os.path.exists(cfg.savepath):
            os.makedirs(cfg.savepath)
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), cfg.savepath + 'model-' + str(epoch + 1) + '.pt')


def union_label(label1, label2):
    C = torch.empty_like(label1)
    for i in range(label1.shape[0]):
        intersection = torch.logical_and(label1[i], label2[i])
        intersection = intersection.float()
        C[i] = intersection
    return C


def Union_box(box1, box2, max_noisy):
    if max_noisy == 0:
        x0 = box2[0, 0].item() - 5
        y0 = box2[0, 1].item() - 5
        x1 = box2[0, 2].item() + 5
        y1 = box2[0, 3].item() + 5
        return torch.tensor([x0, y0, x1, y1], dtype=torch.float).unsqueeze(0)
    else:
        x = abs(box2[0, 2].item() - box2[0, 0].item())
        y = abs(box2[0, 3].item() - box2[0, 1].item())
        if max(x, y) < max_noisy:
            max_noisy /= 2
            x0 = max(box1[0, 0].item() - max_noisy, box2[0, 0].item() - 5)
            y0 = max(box1[0, 1].item() - max_noisy, box2[0, 1].item() - 5)
            x1 = min(box1[0, 2].item() + max_noisy, box2[0, 2].item() + 5)
            y1 = min(box1[0, 3].item() + max_noisy, box2[0, 3].item() + 5)
        else:
            x0 = max(box1[0, 0].item() - max_noisy, box2[0, 0].item() - 5)
            y0 = max(box1[0, 1].item() - max_noisy, box2[0, 1].item() - 5)
            x1 = min(box1[0, 2].item() + max_noisy, box2[0, 2].item() + 5)
            y1 = min(box1[0, 3].item() + max_noisy, box2[0, 3].item() + 5)
        return torch.tensor([x0, y0, x1, y1], dtype=torch.float).unsqueeze(0)


def SIM(out, label):
    # overlap = mask[0, 0] & label
    out = (out >= 0.9).float()
    overlap = out.to(dtype=torch.int) & label.to(dtype=torch.int)
    if torch.count_nonzero(label).item() == 0:
        return 0
    similarity = torch.count_nonzero(overlap).item() / torch.count_nonzero(label).item()
    if similarity <= 0.5:
        return 0
    else:
        return 1


def Filter(mask, label):
    # overlap = mask[0, 0] & label
    overlap = (mask[0, 0] > 0.5).to(dtype=torch.int) & label[0].to(dtype=torch.int)
    if torch.count_nonzero(label[0]).item() == 0:
        return mask, 0
    similarity = torch.count_nonzero(overlap).item() / torch.count_nonzero(label[0]).item()
    if similarity <= 0.5:
        return mask, 0
    else:
        return mask, 1


def val(Dataset, net):
    cfg = Dataset.Config(datapath=Val_PATH, mode='val')
    data = Dataset.Data(cfg)
    val_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    net.train(False)
    dice = 0
    cnt = 0
    prefetcher = DataPrefetcher_val(val_loader)
    image, mask, size, names = prefetcher.next()
    while image is not None:
        with torch.no_grad():
            out2 = net(image, 'Test')
        Dice_is = Dice(out2, mask)
        dice += Dice_is
        cnt += 1
        image, mask, size, names = prefetcher.next()
    print("dice_all =", dice)
    dice /= cnt
    return dice

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)
parser.add_argument(
    "--model-type",
    type=str,
    default="vit_b",
    help="The type of model to load, in ['default', 'vit_l', 'vit_b']",
)

if __name__ == '__main__':
    net = Net()
    args = parser.parse_args()
    args.checkpoint = './checkpoints/sam_vit_b_01ec64.pth'
    train(dataset, net, args)
