# -*- coding: utf-8 -*-
import shutil
import torch

def save_checkpoint(state, is_best, model_dir='models', prefix=None):
    if prefix:
        filename = f'./{model_dir}/{prefix}_model.pth.tar'
        best_file = f'./{model_dir}/{prefix}_model_best.pth.tar'
    else:
        filename = f'./{model_dir}/model.pth.tar'
        best_file = f'./{model_dir}/model_best.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
