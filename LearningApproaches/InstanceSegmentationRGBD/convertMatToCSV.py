# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.06'

import h5py
import csv
import numpy as np
import mxnet as mx
import cv2


def convertfile(filename='/home/smher/Documents/DL_Datasets/NYUv2/nyu_depth_v2_labeled.mat'):
    f = h5py.File(filename)
    images = f['images'][:]     # shape = (1449, 3, 640, 480)
    depths = f['depths'][:]     # shape = (1449, 640, 480)
    labels = f['labels'][:]     # shape = (1449, 640, 380)
    depths = depths[:, np.newaxis, :, :]
    f.close()
    print('shape of images: ', images.shape)
    print('shape of depths: ', depths.shape)
    print('shape of labels: ', labels.shape)
    with open('train_datas_tmp.csv', 'a') as csvfile_img:
        wra = csv.writer(csvfile_img)
        for img, dep in zip(images, depths):
            row = np.concatenate((img.ravel(), dep.ravel()))
            wra.writerow(row)

    with open('train_labels_tmp.csv', 'a') as csvfile_label:
        wrb = csv.writer(csvfile_label)
        for label in labels:
            wrb.writerow(label.ravel())


def test_csv_iter():
   train_img_iter = mx.io.CSVIter(data_csv='train_datas_tmp.csv', data_shape=(4, 640, 480),
                                   label_csv='train_labels_tmp.csv', label_shape=(640, 480),
                                   batch_size=2, dtype='float32')
   for batch in train_img_iter:
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


if __name__ == '__main__':
    test_csv_iter()
    #convertfile()

