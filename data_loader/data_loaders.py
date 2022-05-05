import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
import os, json
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from utils import util


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class TinyImageNetDataloader(BaseDataLoader):
    """
    MiniImageNet data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # 这里首先令训练集与验证集具有相同的trsfm
        trsfm = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.data_dir = data_dir
        self.dataset = TinyImageNetDatasets(data_dir=self.data_dir, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class TinyImageNetDatasets(Dataset):
    def __init__(self, data_dir, train, transform, target_transfrom=None):
        super().__init__()

        image_dir = os.path.join(data_dir, "images")
        assert os.path.exists(image_dir), "dir:'{}' not found.".format(image_dir)

        self.target_transform = target_transfrom
        self.image_folder = image_dir
        self.transform = transform
        self.root = data_dir

        mode = 'train' if train else 'test'
        data_path = os.path.join(data_dir, mode + '_data.pickle')
        target_path = os.path.join(data_dir, mode + '_target.pickle')
        assert os.path.exists(data_path), "file:'{}' not found.".format(data_path)
        assert os.path.exists(target_path), "file:'{}' not found.".format(target_path)

        self.data = util.load_from_pickle(data_path)
        self.targets = util.load_from_pickle(target_path)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.data[item, :], self.targets[item]
