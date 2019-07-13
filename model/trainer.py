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
        for _ in range(self.local_epochs):
            net.train()
        print(f"=> Test begins: Round {rnd}")
        test_acc, test_loss = net.test()
        self.q.put((test_acc, test_loss))

    def model_aggregation(self):
        # compute average of models
        dict_new = collections.defaultdict(list)
        for net in self.nets_pool[:self.num_per_rnd]:
            for k, v in net.model.state_dict().items():
                dict_new[k].append(v)
        for k in dict_new.keys():
            dict_new[k] = torch.mean(torch.stack(dict_new[k]), dim=0)
        return dict_new

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
