#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
import datetime
import random
import collections
import math
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model.agent import Agent
from model.trainer import Trainer, train_local_mp
from model.mnist_models import MNISTModel, MNISTwithAttn
from data.mnist_data import MNISTData

class MNISTAgent(Agent):
    '''
    MNISTAgent for MNIST and Fashion-MNIST.
    '''
    def __init__(self, global_args, subset=tuple(range(10)),
                 fine='MNIST', train_indices=None, test_indices=None):

        super().__init__(global_args, subset, fine, train_indices, test_indices)
        self.permutation = None
        if self.fine == 'MNIST':
            self.permutation = np.random.permutation(28 * 28)

    def load_data(self):
        print("=> loading data")
        self.data = MNISTData(self.subset, self.fine, self.permutation, self.train_indices, self.test_indices)
        self.train_loader = self.data.get_train_loader(self.batch_size, self.num_workers)
        self.test_loader = self.data.get_test_loader(self.batch_size, self.num_workers)

    def build_model(self):
        print("=> building model")
        if self.fusion == 'none':
            self.model = MNISTModel().to(self.device)
        else:
            self.model = MNISTwithAttn(self.fusion).to(self.device)
        if self.fusion in ['multi', 'single']:
            self.shadow = torch.zeros(self.model.attn.gamma.size(), device=self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=5e-4)


class MNISTTrainer(Trainer):
    '''
    MNIST Trainer.
    '''
    def __init__(self, global_args):
        super().__init__(global_args)

        # init the global model
        self.global_agent = MNISTAgent(global_args, fine=self.fine)
        self.global_agent.load_data()
        self.global_agent.build_model()
        self.global_agent.resume_model(self.resume)

    def build_local_models(self, global_args):
        self.nets_pool = list()
        self.nets_pool.append(MNISTAgent(args, subset=[0, 2, 4, 6, 8], fine=args.fine))
        self.nets_pool.append(MNISTAgent(args, subset=[1, 3, 5, 7, 9], fine=args.fine))
        # for i in range(args.num_locals):
        #     nets_pool.append(MNISTAgent(args, fine=args.fine, subset=[2 * i, 2 * i + 1]))
        self.init_local_models()

    def train(self):
        # num_per_rnd = int(0.2 * args.num_locals)
        self.num_per_rnd = 2
        for rnd in range(args.rounds):
            random.shuffle(self.nets_pool)
            for net in self.nets_pool[:self.num_per_rnd]:
                self.train_local(net, rnd)
            self.update_global(rnd)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_trainer = MNISTTrainer(args)

    # test
    if args.mode == 'test':
        mnist_trainer.test()
        return

    writer_dir = os.path.join(f'runs/{args.fine}_{args.fusion}_{args.num_locals}',
                              datetime.datetime.now().strftime('%b%d_%H-%M'))
    mnist_trainer.writer = SummaryWriter(writer_dir)

    mnist_trainer.build_local_models(args)
    mnist_trainer.train()

    mnist_trainer.writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_file',
        type=str,
        default='model.pth.tar',
        help='File to save model.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='train or test.'
    )
    parser.add_argument(
        '--fine',
        type=str,
        default='MNIST',
        choices=('MNIST', 'FashionMNIST'),
        help='Fine choice of dataset.'
    )
    parser.add_argument(
        '--fusion',
        type=str,
        default='none',
        choices=('none', 'multi', 'single', 'conv'),
        help='Method for feature fusion.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='number of workers to preprocess data, must be 0 for mp agents.'
    )
    parser.add_argument(
        '--num_locals',
        type=int,
        default=2,
        help='number of local agents.'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=500,
        help='number of communication rounds.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-3,
        help='learning rate.'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-4,
        help='minimum learning rate.'
    )
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=0.99,
        help='lr decay rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size. (B)'
    )
    parser.add_argument(
        '--local_epochs',
        type=int,
        default=5,
        help='Number of epoch in local. (E)'
    )
    parser.add_argument(
        '--meta_lr',
        type=float,
        default=1e-3,
        help='meta learning rate for model aggregation.'
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to resume checkpoint (default: none)'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory for storing checkpoint file.'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='2',
        help='Number of gpu to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Random seed'
    )
    args = parser.parse_args()
    main()
