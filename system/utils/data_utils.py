# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset

train_transform = transforms.Compose([
    transforms.Resize(336),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])
test_transform = transforms.Compose([
    transforms.Resize(336),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])

#读入npz格式数据集,返回训练集和测试集,格式都是字典{x:(图片个数*256*256*3), y:(对应的标签个数*1)}
def read_data(dataset, data_root,idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(data_root, dataset, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = dict(np.load(f, allow_pickle=True))
        return train_data
    else:
        test_data_dir = os.path.join(data_root, dataset, 'test/')
        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = dict(np.load(f, allow_pickle=True))
        return test_data


def read_client_data(dataset,data_root, idx, is_train=True):
    if is_train:
        #读取数据设定格式，包装为一一对应的sample元组（x,y）,x为图片，y为标签,最后得到一个大小为数据集大小的列表
        train_data = read_data(dataset=dataset, data_root=data_root,idx=idx, is_train=is_train)
        X_train = torch.Tensor(train_data['x']).permute(0, 3, 1, 2) # (N, 256, 256, 3) -> (N, 3, 256, 256)
        X_train = train_transform(X_train).type(torch.float16)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        # transform
        
        #train_data = [(x, y) for x, y in zip(X_train, y_train)]
        train_data= TensorDataset(X_train, y_train)
        return train_data
    else:
        test_data = read_data(dataset=dataset, data_root=data_root,idx=idx, is_train=is_train)
        X_test = torch.Tensor(test_data['x']).permute(0, 3, 1, 2)
        X_test = test_transform(X_test).type(torch.float16)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        #test_data = [(x, y) for x, y in zip(X_test, y_test)]
        test_data = TensorDataset(X_test, y_test)
        return test_data



