# -*- coding: utf-8 -*-
import math
import queue
import collections
from abc import ABC, abstractmethod
import numpy as np
import torch

def train_local_mp(local_epochs, net, rnd, q):
    '''
    For multiprocessing
    '''
    # train
    print(f"=> Train begins: Round {rnd}")
    for _ in range(local_epochs):
        net.train()
    print(f"=> Test begins: Round {rnd}")
    test_acc, test_loss = net.test()
    q.put((test_acc, test_loss))

class Trainer(ABC):
    '''
    Base Trainer class.
    '''
    def __init__(self, global_args):
        self.fine = global_args.fine
        self.num_locals = global_args.num_locals
        self.local_epochs = global_args.local_epochs
        self.rounds = global_args.rounds
        self.fusion = global_args.fusion
        self.policy = global_args.policy
        self.meta_lr = global_args.meta_lr
        self.resume = global_args.resume

        self.writer = None
        self.global_agent = None
        self.nets_pool = None
        self.q = queue.Queue()
        self.num_per_rnd = 0

    def init_local_models(self):
        # duplicate the global model to local nets
        global_state = self.global_agent.model.state_dict()
        for net in self.nets_pool:
            net.load_data()
            net.build_model()
            net.model.load_state_dict(global_state)
            net.model.set_refer(net.device)
        print(f'=> {len(self.nets_pool)} local nets init done.')

    def train_local(self, net, rnd):
        # train
        print(f"=> Train begins: Round {rnd}")
        # start = random.randint(0, 10)
        # net.subset = (tuple(range(10)) * 2)[start:start + 5]
        # net.load_data()
        for _ in range(self.local_epochs):
            net.train()
        print(f"=> Test begins: Round {rnd}")
        test_acc, test_loss = net.test()
        self.q.put((test_acc, test_loss))

    def model_aggregation(self):
        if self.policy == 'average':
            # compute average of models
            dict_new = collections.defaultdict(list)
            for net in self.nets_pool[:self.num_per_rnd]:
                for k, v in net.model.state_dict().items():
                    dict_new[k].append(v)
            for k in dict_new.keys():
                dict_new[k] = torch.mean(torch.stack(dict_new[k]), dim=0)
            return dict_new
        elif self.policy == 'adam':
            local_grads = collections.defaultdict(list)
            global_dict = self.global_agent.model.state_dict()
            for net in self.nets_pool[:self.num_per_rnd]:
                for k, v in net.model.state_dict().items():
                    local_grads[k].append(v)

            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for k, v in global_dict.items():
                grad = v - torch.mean(torch.stack(local_grads[k]), dim=0)
                state = self.global_agent.adam_state[k]
                if not state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = self.meta_lr * math.sqrt(bias_correction2) / bias_correction1

                v.addcdiv_(-step_size, exp_avg, denom)
            return global_dict
        else:
            raise ValueError('Invalid model aggregation policy.')

    def update_global(self, rnd):
        dict_new = self.model_aggregation()

        # update global model and test
        self.global_agent.model.load_state_dict(dict_new)
        self.global_agent.update_lr(rnd)
        if self.fusion in ['multi', 'single']:
            self.global_agent.update_shadow()
        print(f"=> Global Test begins: Round {rnd}")
        self.global_agent.test(rnd)

        dict_new = self.global_agent.model.state_dict()
        # update local models
        for net in self.nets_pool[:self.num_per_rnd]:
            net.model.load_state_dict(dict_new)
            net.model.set_refer(net.device)
            net.update_lr(rnd)
        local_test = list()
        while not self.q.empty():
            local_test.append(self.q.get())
        local_acc, local_loss = np.mean(np.asarray(local_test), axis=0)
        self.writer.add_scalar('local/accuracy', local_acc, rnd)
        self.writer.add_scalar('local/loss', local_loss, rnd)
        self.global_agent.maybe_save(rnd, local_acc)

    def test(self):
        print("=> Test begins.")
        self.global_agent.test()

    @abstractmethod
    def build_local_models(self, global_args):
        pass

    @abstractmethod
    def train(self):
        pass
