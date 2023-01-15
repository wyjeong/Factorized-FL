import os
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from modules.layers import *
import torchvision

class Head(nn.Module):
    def __init__(self, n_clss, factorize=False, hadamard=False, l1=5e-4, gamma=0.1, rank=1):
        super().__init__()
        self.feat_dim = 256 #if self.args.backbone == 'resnet9' else 512 
        if factorize:
            self.head = FactorizedDense(self.feat_dim, n_clss, l1=l1, rank=rank)
        else:
            self.head = nn.Linear(self.feat_dim, n_clss)
        
    def forward(self, x):
        return self.head(x)

class ResNet9(nn.Module):
    def __init__(self, factorize=False, hadamard=False, l1=5e-4, gamma=0.1, rank=1):
        super().__init__()
        self.factorize = factorize
        self.hadamard = hadamard
        self.l1 = l1
        self.rank = rank
        self.gamma = gamma
        self.encoder = nn.Sequential(
            self.conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            self.conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(nn.Sequential(
                self.conv_bn(128, 128),
                self.conv_bn(128, 128),
            )),
            self.conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            Residual(nn.Sequential(
                self.conv_bn(256, 256),
                self.conv_bn(256, 256),
            )),
            self.conv_bn(256, 256, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
        )
        
    def conv_bn(self, channels_in, channels_out, kernel_size=3, stride=1, 
                        padding=1, groups=1, bn=True, activation=True):
        if self.factorize:
            op = [FactorizedConv(channels_in, channels_out, kernel_size, 
                        stride=stride, padding=padding, groups=groups, bias=False, l1=self.l1, rank=self.rank)]
        else:
            op = [nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, 
                        stride=stride, padding=padding, groups=groups, bias=False)]
        if bn:
            op.append(nn.BatchNorm2d(channels_out))
            # op.append(nn.GroupNorm(16, channels_out))
        if activation:
            op.append(nn.ReLU(inplace=True))
        return nn.Sequential(*op)

    def forward(self, x):
        x = self.encoder(x)
        return x

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

