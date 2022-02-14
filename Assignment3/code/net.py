import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from modules import E_ResNet18, D, MFF, R
from torchvision import utils


class Net(nn.Module) :
    def __init__(self) :
        super(Net,self).__init__()
        self.E = E_ResNet18()
        self.D = D()
        self.MFF = MFF()
        self.R = R()
    
    def forward(self, x) :
        x_block1, x_block2, x_block3, x_block4 = self.E(x) #encode module
        x_D = self.D(x_block4) #decode module
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4) #mff module
        x_Rin = torch.cat((x_D, x_mff), 1) #concentate mff and decode result
        x_Rout = self.R(x_Rin) # refinement module

        return x_Rout