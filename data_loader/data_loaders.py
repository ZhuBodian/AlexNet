import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import utils
from base import BaseDataLoader
import os, json
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from utils import util
from sklearn.model_selection import StratifiedShuffleSplit


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
        a = 1


class TinyImageNetDataloader(BaseDataLoader):
    """
    MiniImageNet data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 assign_val_sample=False):
        # 这里首先令训练集与验证集具有相同的trsfm
        trsfm = {
            "train": transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize([224,224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        self.data_dir = data_dir
        self.dataset = TinyImageNetDatasets(path=self.data_dir, train=training, transform=trsfm, split=validation_split)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, assigned_val=assign_val_sample, samplers=self.dataset.samples)


class TinyImageNetDatasets(Dataset):
    def __init__(self, path, train, transform, split):
        super().__init__()

        image_dir = os.path.join(path, "images")
        assert os.path.exists(image_dir), "dir:'{}' not found.".format(image_dir)

        self.target_transform = None
        self.image_folder = image_dir
        self.transform = transform
        self.root = path

        # 根据布尔值train，来确定是生成训练集数据（如果是训练集，那么肯定也要生成验证集），还是测试集数据
        csv_mode = 'train' if train else 'test'
        csv_name = csv_mode + '.csv'
        csv_path = os.path.join(path, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found. please run creat_tiny.py firstly".format(csv_path)
        csv_data = pd.read_csv(csv_path)

        csv_image_name_list = list(csv_data['filename'])
        csv_image_label_list = list(set(csv_data['label']))  # set具有唯一性，但是无序
        csv_image_label_list.sort()  # 注意list的sort函数为就地sort

        # 生成self.target
        for id, image_label in enumerate(csv_image_label_list):
            csv_data.replace(image_label, id, inplace=True)
        self.targets = torch.from_numpy(np.array(csv_data['label'])).long()[:300]

        self.samples = self.cal_samples(split, self.targets)

        """之后的任务：探究不同数据增强对图片的影响，对训练结果的影响，并对pytorch-template进行修改，添加事先分隔号训练集、验证集的功能"""
        data_list = []
        for idx, image_name in enumerate(csv_image_name_list[:300]):
            image_path = os.path.join(os.getcwd(), image_dir, image_name)
            # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
            image = Image.open(image_path).convert('RGB')
            image = transform['train'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）

            data_list.append(image)

            ############################
            # temp_trsfm = transforms.Compose([transforms.Resize(224)
            #                                  transforms.RandomHorizontalFlip(),
            #                                  transforms.ToTensor()])
            # image_path = os.path.join(os.getcwd(), image_dir, image_name)
            # image = Image.open(image_path).convert('RGB')
            # image = temp_trsfm(image)
            # transforms.ToPILImage()(image).show()

        self.data = torch.stack(data_list)
        del data_list

        num_label2text_label = dict([(idx, text) for idx, text in enumerate(csv_image_label_list)])
        # util.write_json(num_label2text_label, os.path.join(path, 'num_label2text_label.json'))

    def cal_samples(self, rate, targets):
        """事先分割好训练集与验证集，方便后续不同的transform"""
        split = StratifiedShuffleSplit(n_splits=1, test_size=rate, random_state=42)
        for train_index, valid_index in split.split(np.zeros(len(targets)), np.array(targets)):
            train_sampler, valid_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(valid_index)
        return [train_sampler, valid_sampler]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.data[item, :], self.targets[item]
