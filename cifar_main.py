#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import datetime
import random
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from model.agent import Agent
from model.trainer import Trainer, train_local_mp
from model.cifar_models import CifarModel, CifarwithAttn
from data.cifar_data import CifarData


class CIFARAgent(Agent):
    '''
    CIFARAgent for CIFAR10 and CIFAR100.
    '''
    def __init__(self, global_args, subset=tuple(range(10)),
                 fine='CIFAR10', train_indices=None, test_indices=None):
        super().__init__(global_args, subset, fine, train_indices, test_indices)

    def load_data(self):
        print("=> loading data")
        self.data = CifarData(self.subset, self.fine, self.train_indices, self.test_indices)
        self.train_loader = self.data.get_train_loader(self.batch_size, self.num_workers)
        self.test_loader = self.data.get_test_loader(self.batch_size, self.num_workers)

    def build_model(self):
        print("=> building model")
        if self.fine == 'CIFAR10':
            num_class = 10
        elif self.fine == 'CIFAR100':
            num_class = 100
        else:
            raise ValueError('Invalid dataset choice.')
        if self.fusion == 'none':
            self.model = CifarModel(num_class).to(self.device)
        else:
            self.model = CifarwithAttn(self.fusion, num_class).to(self.device)
        if self.fusion in ['multi', 'single']:
            self.shadow = torch.zeros(self.model.attn.gamma.size(), device=self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=5e-4)


class CIFARTrainer(Trainer):
    '''
    CIFAR Trainer.
    '''
    def __init__(self, global_args):
        super().__init__(global_args)

        # init the global model
        self.global_agent = CIFARAgent(global_args, fine=self.fine)
        self.global_agent.load_data()
        self.global_agent.build_model()
        self.global_agent.resume_model(self.resume)

    def build_local_models(self, global_args):
        self.nets_pool = list()
        train_indices = np.random.permutation(50000)
        test_indices = np.random.permutation(10000)
        train_per_local = len(train_indices) // self.num_locals
        test_per_local = len(test_indices) // self.num_locals
        for i in range(self.num_locals):
            t_train_indices = train_indices[i * train_per_local: (i + 1) * train_per_local]
            t_test_indices = test_indices[i * test_per_local: (i + 1) * test_per_local]
            self.nets_pool.append(CIFARAgent(global_args, fine=self.fine, subset=tuple(range(10)),
                train_indices=t_train_indices, test_indices=t_test_indices))
        self.init_local_models()

    def train(self):
        ctx = mp.get_context('forkserver')
        self.num_per_rnd = 5
        for rnd in range(self.rounds):
            random.shuffle(self.nets_pool)
            pool = ctx.Pool(self.num_per_rnd)
            self.q = ctx.Manager().Queue()
            pool.starmap(train_local_mp, [(self.local_epochs, net, rnd, self.q) for net in self.nets_pool[:self.num_per_rnd]])
            pool.close()
            pool.join()
            self.update_global(rnd)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cifar_trainer = CIFARTrainer(args)

    # test
    if args.mode == 'test':
        cifar_trainer.test()
        return

    writer_dir = os.path.join(f'runs/{args.fine}_adam_{args.num_locals}',
                              datetime.datetime.now().strftime('%b%d_%H-%M'))
    cifar_trainer.writer = SummaryWriter(writer_dir)

    cifar_trainer.build_local_models(args)
    cifar_trainer.train()

    cifar_trainer.writer.close()

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
        default='CIFAR10',
        choices=('CIFAR10', 'CIFAR100'),
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
        default=0,
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
        default=5e-3,
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
