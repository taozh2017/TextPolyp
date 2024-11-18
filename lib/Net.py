import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Res2Net_v1b import res2net50_v1b_26w_4s


# from .ResNet import ResNet
# from utils.tensor_ops import cus_sample, upsample_add


class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))

    def forward(self, x1, x2, x3, x4):
        x2, x3, x4 = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear')
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear')
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear')
        out = x2 * x3 * x4
        return out


class Net(nn.Module):

    def __init__(self, channel=64):
        super(Net, self).__init__()
        self.bkbone = res2net50_v1b_26w_4s(pretrained=True)
        channels = [256, 512, 1024, 2048]
        self.fusion = Fusion(channels)
        self.linear = nn.Conv2d(channel, 1, kernel_size=1)

    def forward(self, x, mode=None):
        x = self.bkbone.conv1(x)
        x = self.bkbone.bn1(x)
        x = self.bkbone.relu(x)
        x0 = self.bkbone.maxpool(x)
        x1 = self.bkbone.layer1(x0)
        x2 = self.bkbone.layer2(x1)
        x3 = self.bkbone.layer3(x2)
        x4 = self.bkbone.layer4(x3)
        pred = self.fusion(x1, x2, x3, x4)
        pred = self.linear(pred)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)
        if mode == 'train':
            pred = torch.cat((1 - pred, pred), 1)
        return pred


if __name__ == '__main__':
    ras = Net().cuda()
    input_tensor = torch.randn(2, 3, 320, 320).cuda()
    out = ras(input_tensor)
