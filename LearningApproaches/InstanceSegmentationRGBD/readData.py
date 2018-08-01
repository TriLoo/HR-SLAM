# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.14'

import numpy as np
from mxnet import nd
from mxnet import gluon
from mxnet import image
#import scipy.io as scio    # please use HDF reader for matlab v7.3 files
import h5py
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

import joblib
import os

'''
Dataset:  
    NYU Dataset V2
Method:
    Input: .mat file
    Output: Data set of mxnet
    Tool: scipy.io
'''

mat_dir = '/home/smher/Documents/DL_Datasets/NYUv2/nyu_depth_v2_labeled.mat'


def calculate_means_std(datas, means_file='means.joblib', std_file='std.joblib'):
    means = np.mean(datas, axis=(0, 1, 2))   # channel wise
    stds = np.std(datas, axis=(0, 1, 2))
    joblib.dump(means, means_file)
    joblib.dump(stds, std_file)


def calculate_means(datas, means_file='means.joblib'):
    means = np.mean(datas, axis=(0, 1, 2))   # channel wise
    joblib.dump(means, means_file)
    return means


def calculate_std(datas, std_file='std.joblib'):
    stds = np.std(datas, axis=(0, 1, 2))
    joblib.dump(stds, std_file)
    return stds


def rand_crop(data, label, shape):
    data, rect = image.random_crop(data, shape)
    label = image.fixed_crop(label, *rect)

    return data, label


# 变换颜色，增加深度信息与颜色信息之间鲁棒性
augs = gluon.data.vision.transforms.Compose(
    [
        gluon.data.vision.transforms.RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),     # input tensor with (H, W, C)
    ]
)


def normalize_img(img, img_mean, img_std):
    data = (img.astype('float32')/255.0 - img_mean) / img_std
    return data


class NYUDataset(gluon.data.Dataset):
    def __init__(self, filename, mean_file='means.joblib', std_file='std.joblib', **kwargs):
        super(NYUDataset, self).__init__(**kwargs)
        f = h5py.File(filename)
        #print(list(f))   # refs, subsymtem, accelData, depths, images, instances, labels, names, namesToIds, rawDepthFilenames, rawDeths ...
        #print(f['depths'].shape)     # (1449, 640, 480)
        #print(f['images'].shape)     # (1449, 3, 640, 480)
        #print(f['labels'].shape)     # (1449, 640, 480)
        images = f['images']        # type h5py._h1.dataset.Dataset, element is np.ndarray, 下同
        depths = f['depths']
        depths = depths[:, :, :, np.newaxis]
        images = np.transpose(images, axes=(0, 2, 3, 1))
        if os.path.exists(mean_file):
            mean_val = joblib.load(mean_file)
        else:
            mean_val = calculate_means(images)
        if os.path.exists(std_file):
            std_val = joblib.load(std_file)
        else:
            std_val = calculate_std(images)
        self.images = [normalize_img(image, mean_val, std_val) for image in images]
        depth_mean = calculate_means(depths)
        depth_std = calculate_std(depths)
        self.depths = [normalize_img(depth, depth_mean, depth_std) for depth in depths]
        self.labels = f['labels']
        f.close()

    def __getitem__(self, item):    # 可以加入一些数据处理的函数
        return self.images[item], self.depths[item], self.labels[item]

    def __len__(self):
        return len(self.images)


# For Test
if __name__ == '__main__':
    trainset = NYUDataset(mat_dir)
    train_data = gluon.data.DataLoader(trainset, batch_size=1)
    for img0, depth0, label0 in train_data:
        plt.imshow(img0)
        plt.figure()
        plt.imshow(depth0, cmap='gray')
        plt.figure()
        plt.imshow(label0, cmap='gray')
        plt.show()
        break
