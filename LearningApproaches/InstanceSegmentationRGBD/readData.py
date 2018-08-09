# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.14'

import numpy as np
import cv2
import mxnet as mx
from mxnet import gluon
from mxnet import image
import argparse

'''
Dataset:  
    NYU Dataset V2
Method:
    Input: .mat file
    Output: Data set of mxnet
    Tool: scipy.io
'''

# TODO: Need more aggressive augmentations

mat_dir = '/home/smher/Documents/DL_Datasets/NYUv2/nyu_depth_v2_labeled.mat'


# data includs both images and depths
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


'''
class NYUDataset(gluon.data.Dataset):
    def __init__(self, filename, mean_file='means.joblib', std_file='std.joblib', **kwargs):
        super(NYUDataset, self).__init__(**kwargs)

    def __getitem__(self, item):    # 可以加入一些数据处理的函数
        pass

    def __len__(self):
        pass
'''


def get_nyu_iterator():
    #parse = argparse.ArgumentParser()
    #parse.add_argument('--data_csv_file', default='train_datas_tmp.csv')
    #parse.add_argument('--label_csv_file', default='train_labels_tmp.csv')
    #parse.add_argument('--mean_img_file', default='nyu_image_means.joblib')
    #parse.add_argument('--std_img_file', default='nyu_image_stds.joblib')
    #parse.add_argument('--mean_dep_file', default='nyu_depth_means.joblib')
    #parse.add_argument('--std_dep_file', default='nyu_depth_stds.joblib')
    #parse.add_argument('--batch_size', type=int, default=2)
    #args = parse.parse_args()

    train_img_iter = mx.io.CSVIter(data_csv='train_datas_tmp.csv', data_shape=(4, 640, 480),
                               label_csv='train_labels_tmp.csv', label_shape=(640, 480),
                               batch_size=2, dtype='float32')

    return train_img_iter

# For Test
if __name__ == '__main__':
    train_iter = get_nyu_iterator()
    for batch in train_iter:
        print(batch.data[0].shape, batch.label[0].shape)
        img = batch.data[0][:, :3, :, :]    # shape: (2, 3, 640, 480)
        dep = batch.data[0][:, 3:, :, :]     # shape: (2, 1, 640, 480)
        print('shape of img: ', img.shape)
        print('shape of dep: ', dep.shape)
        img_n = img[0]
        img_n = np.transpose(img_n, axes=(1, 2, 0))
        print('shape of img_n: ', img_n.shape)
        print('data type of img_n: ', img_n.dtype)
        print('type of img_n: ', type(img_n))    # mx.ndarray.NDArray
        assert img_n.dtype == np.float32
        img_n = img_n.asnumpy()
        img_n /= np.max(img_n)
        print('type of img_n: ', type(img_n))    # mx.ndarray.NDArray
        assert isinstance(img_n, np.ndarray)
        cv2.imshow('image', img_n)
        dep_n = dep[0][0].asnumpy()
        dep_n /= np.max(dep_n)
        cv2.imshow('depth', dep_n)
        # show label
        label0 = batch.label[0][0]   # (640, 480)
        label0 = label0.asnumpy()
        print('data type of label0: ', label0.dtype)
        print('max of label0: ', np.max(label0))
        label0 /= np.max(label0)
        cv2.imshow('label', label0)
        cv2.waitKey()
        break
