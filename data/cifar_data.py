# -*- coding: utf-8 -*-
import os
import sys
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

class CIFAR10(torch.utils.data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        subset(list of class ids, optional): Sample a subset of CIFAR10.

    """
    base_folder = 'cifar-10-batches-py'
    filename = "cifar-10-python.tar.gz"
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
    }

    def __init__(self, root, train=True, subset=tuple(range(10)),
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                labels = None
                if 'labels' in entry:
                    labels = entry['labels']
                else:
                    labels = entry['fine_labels']
                for k, v in zip(entry['data'], labels):
                    if v not in subset:
                        continue
                    self.data.append(k)
                    self.targets.append(v)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    filename = "cifar-100-python.tar.gz"
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
    }


class CifarData():
    '''
    Load cifar data with samplers.
    '''
    def __init__(self, subset=tuple(range(10)), fine='CIFAR10', train_indices=None, test_indices=None):
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_sampler = None
        self.test_sampler = None
        if fine == 'CIFAR10':
            data_dir = 'dataset/cifar10'
        elif fine == 'CIFAR100':
            data_dir = 'dataset/cifar100'
        else:
            raise ValueError('Invalid dataset choice.')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                          std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.trainset = globals()[fine](
            root=data_dir,
            train=True,
            subset=subset,
            transform=train_transform)
        self.testset = globals()[fine](
            root=data_dir,
            train=False,
            subset=subset,
            transform=test_transform)

        if self.train_indices is not None:
            self.train_sampler = SubsetRandomSampler(self.train_indices)
        if self.test_indices is not None:
            self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_loader(self, batch_size, num_workers):
        train_loader = DataLoader(self.trainset, batch_size=batch_size,
                                  shuffle=(self.train_sampler is None),
                                  sampler=self.train_sampler,
                                  num_workers=num_workers, pin_memory=True)
        return train_loader

    def get_test_loader(self, batch_size, num_workers):
        test_loader = DataLoader(self.testset, batch_size=batch_size,
                                 shuffle=False, sampler=self.test_sampler,
                                 num_workers=num_workers, pin_memory=True)
        return test_loader

def main():
    cifar_data = CifarData(fine='CIFAR100', subset=tuple(range(100)))
    print(len(cifar_data.trainset), len(cifar_data.testset))
    # train_loader = cifar_data.get_train_loader(batch_size=128, num_workers=8)
    # print(len(train_loader))

if __name__ == '__main__':
    main()
