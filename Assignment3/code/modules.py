import torch
from torch.nn.modules.activation import ReLU
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from blocks import ResBlock, UpProjection

#Encoder using Resnet18
class E_ResNet18(nn.Module):
    def __init__(self, num_classes = 1000):
        super(E_ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, 2),
            ResBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, 2),
            ResBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, 2),
            ResBlock(512, 512, 1)
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x) :
        x = self.conv1(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

#Decoder
class D(nn.Module) :
    def __init__(self, in_channel = 512):
        super(D, self).__init__()
        out_channel = in_channel//2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        in_channel = out_channel
        out_channel = in_channel//2
        self.up1 = UpProjection(in_channel,out_channel)

        in_channel = out_channel
        out_channel = in_channel//2
        self.up2 = UpProjection(in_channel,out_channel)

        in_channel = out_channel
        out_channel = in_channel//2
        self.up3 = UpProjection(in_channel,out_channel)

        in_channel = out_channel
        out_channel = in_channel//2
        self.up4 = UpProjection(in_channel,out_channel)
    
    def forward(self, x) :
        x_0 = self.conv2(x)
        x_1 = self.up1(x_0, [15,19])
        x_2 = self.up2(x_1, [29,38])
        x_3 = self.up3(x_2, [57,76])
        x_4 = self.up4(x_3, [114,152])

        return x_4

#Multi-scale Feature Fusion Module
class MFF(nn.Module) :
    def __init__(self, in_channel = 64):
        super(MFF, self).__init__()

        self.up5 = UpProjection(64, 16) #block1 channel 64
        self.up6 = UpProjection(128, 16) #block2 channel 128
        self.up7 = UpProjection(256, 16) #block3 channel 256
        self.up8 = UpProjection(512, 16) #block4 channel 512
        out_channel = in_channel
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x_block1, x_block2, x_block3, x_block4, size=[114, 152]) :
        x_up1 = self.up5(x_block1,size)
        x_up2 = self.up6(x_block2,size)
        x_up3 = self.up7(x_block3,size)
        x_up4 = self.up8(x_block4,size)

        x_all = torch.cat((x_up1, x_up2, x_up3, x_up4), 1)
        out = self.conv3(x_all)

        return out

#Refinement Module
class R(nn.Module) :
    def __init__(self, in_channel = 80): # channel num = 64(MFF) + 16(D) = 80
        super(R,self).__init__()
        out_channel = in_channel

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

        self.conv6 = nn.Conv2d(in_channel, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x) :
        out = self.conv4(x)
        out = self.conv5(out)
        out = self.conv6(out)

        return out

