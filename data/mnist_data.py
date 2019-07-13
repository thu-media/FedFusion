# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms


class MNIST(torch.utils.data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, subset=tuple(range(10)), permutation=None,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.permutation = permutation # for permutation mnist

        if not self._check_exists():
            print(self.root)
            raise RuntimeError('Dataset not found.')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = list(), list()
        total_data, total_targets = torch.load(os.path.join(self.processed_folder, data_file))
        for k, v in zip(total_data, total_targets):
            if int(v) in subset:
                self.data.append(k)
                self.targets.append(v)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.permutation is not None:
            img = img.view(1, -1)[:, self.permutation].reshape(1, 28, 28)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

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

class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class MNISTData():
    '''
    Load mnist data with samplers.
    '''
    def __init__(self, subset=tuple(range(10)), fine='MNIST', permutation=None,
                 train_indices=None, test_indices=None):
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_sampler = None
        self.test_sampler = None
        if fine == 'MNIST':
            data_dir = 'dataset/mnist'
            normalize = transforms.Normalize(mean=[0.131], std=[0.308])
        elif fine == 'FashionMNIST':
            data_dir = 'dataset/fashion_mnist'
            normalize = transforms.Normalize(mean=[0.286], std=[0.353])
        else:
            raise ValueError('Invalid dataset choice.')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(28),
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
            permutation=permutation,
            transform=train_transform)
        self.testset = globals()[fine](
            root=data_dir,
            train=False,
            subset=subset,
            permutation=permutation,
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

if __name__ == "__main__":
    # mnist = globals()['MNIST']('dataset/mnist')
    # fashion = globals()['FashionMNIST']('dataset/fashion_mnist')
    # print(mnist)
    # print(fashion)
    data = MNISTData(fine='FashionMNIST')
    print(data.trainset)
    print(data.testset)
