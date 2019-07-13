# -*- coding: utf-8 -*-
import os
import time
import collections
from abc import ABC, abstractmethod
import torch

from . import utils

class Agent(ABC):
    '''
    Base Agent.
    '''
    def __init__(self, global_args, subset, fine, train_indices, test_indices):
        self.best_acc = 0.
        self.test_acc = 0.
        self.test_loss = 0.
        self.subset = subset
        self.fine = fine
        self.train_indices = train_indices
        self.test_indices = test_indices
        # self.device = torch.device(f'cuda:{args.gpu}')
        self.device = torch.device('cuda:0')
        self.fusion = global_args.fusion

        self.lr = global_args.lr
        self.min_lr = global_args.min_lr
        self.decay_rate = global_args.decay_rate
        self.batch_size = global_args.batch_size
        self.num_workers = global_args.num_workers
        self.adam_state = collections.defaultdict(dict)

        self.model_dir = global_args.model_dir

    @abstractmethod
    def load_data(self):
        print("=> loading data")
        self.data = None
        self.train_loader = None
        self.test_loader = None

    @abstractmethod
    def build_model(self):
        print("=> building model")
        self.model = None
        self.shadow = None
        self.criterion = None
        self.optimizer = None

    def resume_model(self, resume_path):
        # optionally resume from a checkpoint
        if resume_path:
            resume_path = f'models/{resume_path}.pth.tar'
            if os.path.isfile(resume_path):
                print(f"=> loading checkpoint '{resume_path}'")
                checkpoint = torch.load(resume_path, map_location=self.device)
                self.best_acc = checkpoint['best_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                print(f"=> loaded checkpoint '{resume_path}' (Round {checkpoint['rnd']})")
                del checkpoint
            else:
                print(f"=> no checkpoint found at '{resume_path}'")

    def test(self, rnd=None, writer=None):
        batch_time = utils.AverageMeter()
        loss_meter = utils.AverageMeter()
        top1 = utils.AverageMeter()
        # switch to eval mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for image_batch, label_batch in self.test_loader:
                image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)

                # compute output
                output, _ = self.model(image_batch)
                loss = self.criterion(output, label_batch)
                loss_meter.update(loss.item(), label_batch.size(0))

                # measure accuracy
                acc, *_ = utils.accuracy(output, label_batch)
                top1.update(acc[0].item(), label_batch.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            print(f' * Accuracy {top1.avg:.3f}')
            if rnd and writer is not None:
                writer.add_scalar('global/loss', loss_meter.avg, rnd)
                writer.add_scalar('global/accuracy', top1.avg, rnd)
                if self.fusion in ['multi', 'single']:
                    writer.add_scalar('global/gamma', self.model.attn.gamma.mean().item(), rnd)

        self.test_acc = top1.avg
        self.test_loss = loss_meter.avg
        return self.test_acc, self.test_loss

    def train(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        loss_meter = utils.AverageMeter()
        top1 = utils.AverageMeter()
        # switch to train mode
        self.model.train()

        end = time.time()
        for image_batch, label_batch in self.train_loader:
            # measure data loading time
            data_time.update(time.time() - end)
            image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)

            # compute output and loss
            output, _ = self.model(image_batch)
            loss = self.criterion(output, label_batch)
            loss_meter.update(loss.item(), label_batch.size(0))

            # measure accuracy
            acc, *_ = utils.accuracy(output, label_batch)
            top1.update(acc[0].item(), label_batch.size(0))

            # compute gradient and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    def update_lr(self, rnd, writer=None):
        if writer is not None:
            writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], rnd)
        if self.lr > self.min_lr:
            self.lr *= self.decay_rate
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr

    def maybe_save(self, rnd, local_acc):
        is_best = local_acc > self.best_acc
        if is_best:
            self.best_acc = local_acc

        utils.save_checkpoint({
            'rnd': rnd + 1,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            # 'optimizer' : self.optimizer.state_dict(),
        }, is_best, model_dir=self.model_dir, prefix=self.fine)

    def update_shadow(self, mu=0.9):
        gamma = self.model.attn.gamma
        gamma.data = (1 - mu) * gamma + mu * self.shadow
        self.shadow = gamma.data.clone()
