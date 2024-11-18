import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def SaliencyStructureConsistency(x, y, alpha):
    ssim = torch.mean(SSIM(x, y))
    l1_loss = torch.mean(torch.abs(x - y))
    loss_ssc = alpha * ssim + (1 - alpha) * l1_loss
    return loss_ssc


def uncertainty(label1, label2):
    label_p = 0.5 * (label1 + label2)
    p1 = abs(label_p - label1)
    p2 = abs(label_p - label2)
    return torch.exp(-p1), torch.exp(-p2)


def structure_loss(pred, mask, weit):
    wbce = F.binary_cross_entropy(pred, mask, reduce=None, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)
