# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.14 -> 2018.08.16'

import numpy as np
from mxnet import nd
import cv2
import mxnet as mx
from mxnet import gluon
from mxnet import image
from collections import Iterator

'''@package docstring
Customed CSVIter to add image augmentations.

'''

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
nyu_augs = gluon.data.vision.transforms.Compose(
    [
        gluon.data.vision.transforms.RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)     # input tensor with (H, W, C)
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


# 先用这个吧，先把程序跑起来(2018.08.14)
def get_nyu_iterator(data_file='train_datas_tmp.csv',
                     data_shape=(4, 640, 480),
                     label_file='train_labels_tmp.csv',
                     label_shape=(640, 480),
                     batch_size=2):
    train_img_iter = mx.io.CSVIter(data_csv=data_file, data_shape=data_shape,
                                   label_csv=label_file, label_shape=label_shape,
                                   batch_size=batch_size, dtype='float32')

    return train_img_iter


## The implementation of customed data iterator.
#
# Should override *__next()__* in python3, returnning a `DataBatch` or raising a
# `StopIteration` exception if at the end of the data stream.
#
# Should override `reset()` method to restart reading from the beginning
#
# Have a `provide_data` attribute, consisting of a list of `DataDesc` objects that
# store the name, shape, type and layout information of the data.
#
# Have a `provide_label` providing above information of the label.
#
# add `augs` into the parameter list of this class
class myCSVIter(mx.io.DataIter):
    def __init__(self,
                 data_file='train_datas_tmp.csv',
                 data_shape=(4, 640, 480),
                 label_file='train_labels_tmp.csv',
                 label_shape=(640, 480),
                 batch_size=2,
                 augs=None,
                 **kwargs):
        super(myCSVIter, self).__init__(kwargs)
        self.csv_iter = get_nyu_iterator(data_file, data_shape, label_file, label_shape, batch_size)
        self._provide_data = self.csv_iter.provide_data
        self._provide_label = self.csv_iter.provide_label
        self._augs = augs

    def reset(self):
        self.csv_iter.reset()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def __next__(self):
        return self.next()

    def next(self):     # Cannot convert list of ndarray to a new ndarray
        try:
            databatch_ = self.csv_iter.next()
            img_data = databatch_.data[0]
            img_label = databatch_.label[0]
            img_rgb = np.transpose(img_data[:, :3, :, :], axes=(0, 2, 3, 1))
            img_lst = [self._augs(img).asnumpy() for img in img_rgb]
            img = nd.array(img_lst)
            img = nd.transpose(img, axes=(0, 3, 1, 2))
            img_data[:, :3, :, :] = img
            return mx.io.DataBatch([img_data], [img_label])
        except StopIteration:
            raise StopIteration


# For Test
if __name__ == '__main__':
    my_csvIter = myCSVIter(augs=nyu_augs)
    print('type of my_csvIter: ', type(my_csvIter))
    print('provide_data of my_csvIter: ', my_csvIter.provide_data)
    print('provide_label of train_iter: ', my_csvIter.provide_label)
    data_batch = my_csvIter.next()
    print('type of next() return: ', type(data_batch))
    print('shape of data[0]: ', data_batch.data[0].shape)
    print(isinstance(my_csvIter, Iterator))
    i = 0
    for batch in my_csvIter:
        print('type of batch: ', type(batch))
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
        print('maximum of label0: ', np.max(label0))
        print('minimum of label0: ', np.min(label0))
        label0 /= np.max(label0)
        cv2.imshow('label', label0)
        cv2.waitKey()
        i += 1
        if i == 3:
            break
    '''
    train_iter = get_nyu_iterator()
    print('Type of CSVIter instance: ', type(train_iter))    # Note: the type is <class 'mxnet.io.MXDataIter'> ! !
    train_iter.reset()
    print('provide_data of train_iter: ', train_iter.provide_data)
    print('provide_label of train_iter: ', train_iter.provide_label)
    data = train_iter.next()
    print('type of next() return: ', type(data))
    print('shape of data[0]: ', data.data[0].shape)
    '''
