# -*- coding: utf-8 -*-
from itertools import chain
import copy
import torch
from torch import nn
import torch.nn.functional as F

class CifarModel(nn.Module):

    def __init__(self, num_class=10):
        super().__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            # fc3
            nn.Linear(64 * 8 * 8, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # fc4
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # softmax
            nn.Linear(192, num_class),
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

class SelfAttn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()

        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
            inputs :
                x, y: input feature maps (B X C X W X H)
                x from the local model, y from the global one
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batch_size, C, width, height = x.size()
        N = width * height

        proj_query = self.query_conv(y).view(batch_size, -1, N).permute(0, 2, 1) # B X (C X N)^T
        proj_key = self.key_conv(y).view(batch_size, -1, N) # B X C x N
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=2) # B X N X N

        proj_value = self.value_conv(x).view(batch_size, -1, N) # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out, attention

class ActivationAttn(nn.Module):
    """ Activation attention Layer"""
    def __init__(self):
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(1, 64, 1, 1))
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
            inputs :
                x, y: input feature maps (B X C X W X H)
                x from the local model, y from the global one
            returns :
                out : self attention value + input feature
                attention: B X 1 X N (N is Width*Height)
        """
        batch_size, C, width, height = x.size()
        N = width * height

        proj_global = y.pow(2).sum(dim=1).view(batch_size, 1, N) # B X 1 X N
        attention = F.softmax(proj_global, dim=2)
        proj_local = x.view(batch_size, C, N) # B X C x N
        out = torch.mul(attention, proj_local).view(batch_size, C, width, height)
        out = self.gamma * out + proj_local

        return out, attention

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

class CifarwithAttn(CifarModel):
    '''Cifar model with attention components'''
    def __init__(self, policy, num_class=10):
        '''
            policy: which attention component to use.
        '''
        super().__init__(num_class)
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
