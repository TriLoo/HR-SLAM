# coding : utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.14'

import mxnet as mx
from mxnet import nd
from mxnet import gluon
import model
import readData
import cv2
import numpy as np
from collections import Iterator


def data_iter_test():
    my_csvIter = readData.myCSVIter(augs=readData.nyu_augs)
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


def train_model():
    net = model.ASENet(class_nums=13)   # NYUv2: number of classes: 13
    net.initialize(init=mx.init.Xavier())
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1, 'momentum':0.9, 'wd':0.0001})
    epoches = 10
    train_data_iter = readData.myCSVIter(augs=readData.nyu_augs)
    test_data_iter = readData.myCSVIter(data_file='split_test_datas.csv',
                                        label_file='split_test_labels.csv',
                                        batch_size=1)
    model.train(net, trainer, train_data_iter, test_data_iter, epoches, ctx=mx.cpu())


if __name__ == '__main__':
    # test dataset
    # data_iter_test()
    train_model()

