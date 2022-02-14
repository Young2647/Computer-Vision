import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d

class ResBlock(nn.Module) :
    def __init__(self, in_channel, out_channel, stride = 1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if stride != 1 :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding = 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else :
            self.shortcut = nn.Sequential()
        
    def forward(self, x) :
        out = self.layer(x) + self.shortcut(x)
        out = F.relu(out)

        return out

class UpProjection(nn.Module) :
    def __init__(self, in_channel, out_channel):
        super(UpProjection, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
    def forward(self, x, size) :
        x = F.upsample(x, size=size, mode='bilinear')
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)

        out = F.relu(x_1 + x_2)
        return out

