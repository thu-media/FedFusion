# -*- coding: utf-8 -*-
from itertools import chain
import copy
import torch
from torch import nn
import torch.nn.functional as F

class MNISTModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            # fc3
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # softmax
            nn.Linear(512, 10),
        )
        for layer in chain(self.features, self.classifier):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def set_refer(self, device):
        # to keep the api consistent
        pass

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out, x

class FeatureFuse(nn.Module):
    """ Activation attention Layer"""
    def __init__(self, policy):
        super().__init__()
        if policy == 'multi':
            self.gamma = nn.Parameter(torch.zeros(1, 64, 1, 1))
        elif policy == 'single':
            self.gamma = nn.Parameter(torch.zeros(1))
        elif policy == 'conv':
            self.fuse = nn.Conv2d(64 * 2, 64, kernel_size=1)
        else:
            raise ValueError('Invalid attention policy.')

        self.policy = policy

    def forward(self, x, y):
        """
            inputs :
                x, y: input feature maps (B X C X W X H)
                x from the local model, y from the global one
            returns :
                out : fused feature map
        """
        if self.policy in ['multi', 'single']:
            out = self.gamma * y + (1 - self.gamma) * x
        else:
            out = torch.cat((x, y), dim=1)
            out = self.fuse(out)

        return out, None

class MNISTwithAttn(MNISTModel):
    '''MNIST model with attention components'''
    def __init__(self, policy):
        '''
            policy: which attention component to use.
        '''
        super().__init__()
        self.attn = FeatureFuse(policy=policy)
        self.refer_conv = copy.deepcopy(self.features)

    def set_refer(self, device):
        self.refer_conv = copy.deepcopy(self.features).to(device)
        for param in self.refer_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            y = self.refer_conv(x)
        x = self.features(x)
        out, attention = self.attn(x, y)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, attention
