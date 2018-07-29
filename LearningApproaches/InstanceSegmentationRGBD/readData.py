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


# TODO: to be implemented
rgb_mean = nd.array([112, 112, 112])
depth_mean = nd.array([112])
rgb_std = nd.array([1, 1, 1])
depth_std = nd.array([1])


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


def normalize_img(rgb, depth):
    data_rgb = (rgb.astype('float32')/255.0 - rgb_mean) / rgb_std
    data_depth = (depth.astype('float32')/255.0 - depth_mean) / depth_std

    return data_rgb, data_depth


class NYUDataset(gluon.data.Dataset):
    def __init__(self, filename, mean_file='means.joblib', std_file='std.joblib', **kwargs):
        super(NYUDataset, self).__init__(**kwargs)
        mat_file = h5py.File(filename)
        #print(list(f))   # refs, subsymtem, accelData, depths, images, instances, labels, names, namesToIds, rawDepthFilenames, rawDeths ...
        #print(f['depths'].shape)     # (1449, 640, 480)
        #print(f['images'].shape)     # (1449, 3, 640, 480)
        #print(f['labels'].shape)     # (1449, 640, 480)
        images = f['images']        # type h5py._h1.dataset.Dataset, element is np.ndarray, 下同
        depths = f['depths']
        depths = depths[:, :, :, np.newaxis]
        mean_val = joblib.load(mean_file)
        std_val = joblib.load(std_file)
        images = np.transpose(images, axes=(0, 2, 3, 1))
        self.images = [normalize_img(image, depth) for image, depth in zip(images, depths)]
        self.depths = [normalize_img(image, depth) for image, depth in zip(images, depths)]
        self.labels = f['labels']
        f.close()

    def __getitem__(self, item):    # 可以加入一些数据处理的函数
        return self.images[item], self.depths[item], self.labels[item]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    f = h5py.File(mat_dir)
    print(list(f))   # refs, subsymtem, accelData, depths, images, instances, labels, names, namesToIds, rawDepthFilenames, rawDeths ...
    print(f['depths'].shape)     # (1449, 640, 480)
    print(f['images'].shape)    # (1449, 3, 640, 480)
    print(f['labels'].shape)    # (1449, 640, 480)
    images = f['images']        # type h5py._h1.dataset.Dataset, element is np.ndarray, 下同
    depths = f['depths']
    labels = f['labels']
    print('shape of images: ', images.shape)
    img0 = images[0]
    img0 = np.transpose(img0, (1, 2, 0))
    depth0 = depths[0]
    label0 = labels[0]
    f.close()

    plt.imshow(img0)
    plt.figure()
    plt.imshow(depth0, cmap='gray')
    plt.figure()
    plt.imshow(label0, cmap='gray')
    plt.show()

