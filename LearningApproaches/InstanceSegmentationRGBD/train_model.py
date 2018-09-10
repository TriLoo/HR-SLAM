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


def data_iter_test():
    '''
    train_label_iter = mx.io.CSVIter(data_csv='split_train_datas.csv',
                                     data_shape=(4, 640, 480),
                                     label_csv='split_train_labels.csv',
                                     label_shape=(640, 480),
                                     batch_size=1, dtype='float32')
    '''
    test_iter = mx.io.CSVIter(data_csv='split_test_datas.csv',
                              data_shape=(4, 640, 480),
                              label_csv='split_test_labels.csv',
                              label_shape=(640, 480),
                              batch_size=1, dtype='float32')
    i = 0
    for batch in test_iter:
        i += 1
        label = batch.label[0][0, :, :]
        print('shape of label: ', label.shape)
        print('data type of label: ', label.dtype)
        print('type of label: ', type(label))
        print(nd.max(label))
        print(nd.min(label))
        label = label.asnumpy()
        label /= np.max(label)
        label *= 255
        label = label.astype(np.uint8)
        cv2.imshow('test', label)
        img = batch.data[0][:, :3, :, :]    # shape: (1, 3, 640, 480)
        dep = batch.data[0][:, 3:, :, :]     # shape: (1, 1, 640, 480)
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
        cv2.waitKey()
        if i == 3:
            break



def train_model():
    net = model.ASENet(class_nums=40)   # NYUv2: number of classes:
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1, 'momentum':0.9, 'wd':0.0001})
    train_iter = readData.myCSVIter()
    epoches = 10


if __name__ == '__main__':
    # test dataset
    data_iter_test()

