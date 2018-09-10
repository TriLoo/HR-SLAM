# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.27'


import h5py
import numpy as np
import csv
from scipy.io import loadmat
import logging     # Ref: python.jobbole.com/81666
import joblib


#def generate_map_class_13(label_img, map_40, map_13):
def generate_map_class_13(map_40, map_13):
    map_to_13 = []
    map_40 = map_40.astype(np.uint8)    # x: 0 - 893, y: 1 - 40
    map_13 = map_13.astype(np.uint8)    # x: 0 - 39,  y: 1 - 13
    map_to_13.append(0)   # use 0 as background
    for idx in map_40:    # convert [1, 894] to [1, 40]
        map_to_13.append(map_13[idx-1])
    map_to_13 = np.array(map_to_13)    # Convert [1, 894] to [1, 13]
    joblib.dump(map_to_13, 'map_from893_to13.joblib')
    '''
    out_imgs = []
    H, W = label_img[0].shape
    for img in label_img:
        t_img = np.zeros(img.shape)
        for i in range(H):
            for j in range(W):
                t_img[i][j] = map_to_13[img[i][j]]
        out_imgs.append(t_img.astype(np.uint8))
    return np.array(out_imgs)
    '''
    return map_to_13


def map_class_13(label_img, map_13):
    t_img = np.zeros(label_img.shape)
    H, W = t_img.shape
    for i in range(H):
        for j in range(W):
            t_img[i][j] = map_13[label_img[i][j]]
    return t_img


#def generate_csv_dataset_train(input_file, train_idx, test_idx, class_map_40, class_map_13, data_file=None, label_file=None):
def generate_csv_dataset_train(input_file, train_idx, class_map_13, data_file=None, label_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info('Starting split the nyuv2 dataset into train & test .csv files.')
    handler = logging.FileHandler('nyu_split_datsets.log')
    logger.addHandler(handler)
    f = h5py.File(input_file)
    images = f['images'][:]     # shape = (1449, 3, 640, 480)
    depths = f['depths'][:]     # shape = (1449, 640, 480)
    labels = f['labels'][:]     # shape = (1449, 640, 480)
    f.close()
    logger.info('Read images, depths, labels done.')
    print('data tyep of labels: ', labels.dtype)    # uint16
    depths = depths[:, np.newaxis, :, :]
    logger.info('Recast the labels done.')
    print('shape of images: ', images.shape)
    print('shape of depths: ', depths.shape)
    print('shape of labels: ', labels.shape)

    train_img = images[train_idx, :, :, :]
    train_dep = depths[train_idx, :, :, :]
    logger.info('Split train and test done.')

    try:
        with open('split_train_datas.csv', 'a') as csvfile_train_img:
            wra = csv.writer(csvfile_train_img)
            for img, dep in zip(train_img, train_dep):
                row = np.concatenate((img.ravel(), dep.ravel()))
                wra.writerow(row)
    except Exception:
        logger.error('Failed to store train datas.csv', exc_info=True)
    del train_img
    del train_dep

    train_label = labels[train_idx, :, :]
    try:
        with open('split_train_labels.csv', 'a') as csvfile_train_label:
            wrb = csv.writer(csvfile_train_label)
            for label in train_label:
                label = map_class_13(label, class_map_13)   # float64
                wrb.writerow(label.ravel())
    except Exception:
        logger.error('Failed to store train_labels.csv', exc_info=True)


#def generate_csv_dataset_test(input_file, train_idx, test_idx, class_map_40, class_map_13, data_file=None, label_file=None):
def generate_csv_dataset_test(input_file, test_idx, class_map_13, data_file=None, label_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info('Starting split the nyuv2 dataset into train & test .csv files.')
    handler = logging.FileHandler('nyu_split_datsets.log')
    logger.addHandler(handler)
    f = h5py.File(input_file)
    images = f['images'][:]     # shape = (1449, 3, 640, 480)
    depths = f['depths'][:]     # shape = (1449, 640, 480)
    labels = f['labels'][:]     # shape = (1449, 640, 480)
    f.close()
    logger.info('Read images, depths, labels done.')
    print('data tyep of labels: ', labels.dtype)    # uint16
    depths = depths[:, np.newaxis, :, :]
    logger.info('Recast the labels done.')
    print('shape of images: ', images.shape)
    print('shape of depths: ', depths.shape)
    print('shape of labels: ', labels.shape)

    test_img = images[test_idx, :, :, :]
    test_dep = depths[test_idx, :, :, :]
    logger.info('Split train and test done.')

    try:
        with open('split_test_datas.csv', 'a') as csvfile_test_img:
            wrc = csv.writer(csvfile_test_img)
            for img, dep in zip(test_img, test_dep):
                row = np.concatenate((img.ravel(), dep.ravel()))
                wrc.writerow(row)
    except Exception:
        logger.error('Failed to store test_datas.csv', exc_info=True)

    del test_img
    del test_dep
    test_label = labels[test_idx, :, :]
    try:
        with open('split_test_labels.csv', 'a') as csvfile_test_label:
            wrd = csv.writer(csvfile_test_label)
            for label in test_label:
                label = map_class_13(label, map_13=class_map_13)
                wrd.writerow(label.ravel())
    except Exception:
        logger.error('Failed to store test_labels.csv', exc_info=True)

    print('All Done.')


def generate_csv_dataset(input_file, train_idx, test_idx, class_map_40, class_map_13, data_file=None, label_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info('Starting split the nyuv2 dataset into train & test .csv files.')
    handler = logging.FileHandler('nyu_split_datsets.log')
    logger.addHandler(handler)
    f = h5py.File(input_file)
    images = f['images'][:]     # shape = (1449, 3, 640, 480)
    depths = f['depths'][:]     # shape = (1449, 640, 480)
    labels = f['labels'][:]     # shape = (1449, 640, 480)
    f.close()
    logger.info('Read images, depths, labels done.')
    print('data tyep of labels: ', labels.dtype)    # uint16
    labels = map_class_13(labels, class_map_40, class_map_13)
    depths = depths[:, np.newaxis, :, :]
    logger.info('Recast the labels done.')
    print('shape of images: ', images.shape)
    print('shape of depths: ', depths.shape)
    print('shape of labels: ', labels.shape)

    train_img = images[train_idx, :, :, :]
    train_dep = depths[train_idx, :, :, :]
    train_label = labels[train_idx, :, :]
    test_img = images[test_idx, :, :, :]
    test_dep = depths[test_idx, :, :, :]
    test_label = labels[test_idx, :, :]
    logger.info('Split train and test done.')

    try:
        with open('split_train_datas.csv', 'a') as csvfile_train_img:
            wra = csv.writer(csvfile_train_img)
            for img, dep in zip(train_img, train_dep):
                row = np.concatenate((img.ravel(), dep.ravel()))
                wra.writerow(row)
    except Exception:
        logger.error('Failed to store train datas.csv', exc_info=True)

    try:
        with open('split_train_labels.csv', 'a') as csvfile_train_label:
            wrb = csv.writer(csvfile_train_label)
            for label in train_label:
                wrb.writerow(label.ravel())
    except Exception:
        logger.error('Failed to store train_labels.csv', exc_info=True)

    try:
        with open('split_test_datas.csv', 'a') as csvfile_test_img:
            wrc = csv.writer(csvfile_test_img)
            for img, dep in zip(test_img, test_dep):
                row = np.concatenate((img.ravel(), dep.ravel()))
                wrc.writerow(row)
    except Exception:
        logger.error('Failed to store test_datas.csv', exc_info=True)

    try:
        with open('split_test_labels.csv', 'a') as csvfile_test_label:
            wrd = csv.writer(csvfile_test_label)
            for label in test_label:
                wrd.writerow(label.ravel())
    except Exception:
        logger.error('Failed to store test_labels.csv', exc_info=True)

    print('All Done.')


if __name__ == '__main__':
    file_name = '/home/smher/Documents/DL_Datasets/NYUv2/nyu_depth_v2_labeled.mat'
    idx_file = '/home/smher/Documents/DL_Datasets/NYUv2/nyuv2-meta-data/splits.mat'
    class40_file = '/home/smher/Documents/DL_Datasets/NYUv2/nyuv2-meta-data/classMapping40.mat'
    class13_file = '/home/smher/Documents/DL_Datasets/NYUv2/nyuv2-meta-data/classMapping40_13.mat'
    idx = loadmat(idx_file)
    print(list(idx))
    train_idx = idx['trainNdxs']
    test_idx = idx['testNdxs']
    print('type of train_idx: ', type(train_idx))      # numpy.ndarray
    print('shape of train_idx: ', train_idx.shape)     # (795, 1)
    train_idx = np.reshape(train_idx, newshape=-1)
    test_idx = np.reshape(test_idx, newshape=-1)
    train_idx -= 1
    test_idx -= 1
    class40_mat = loadmat(class40_file)
    class13_mat = loadmat(class13_file)
    print(list(class40_mat))
    print(list(class13_mat))
    class_map_40 = class40_mat['mapClass']
    class_map_13 = class13_mat['mapClass']
    class_map_40 = np.reshape(class_map_40, newshape=-1)
    class_map_13 = np.reshape(class_map_13, newshape=-1)
    class_map_to_13 = generate_map_class_13(class_map_40, class_map_13)

    print('Starting.')
    #generate_csv_dataset_train(input_file=file_name, train_idx=train_idx, test_idx=test_idx, class_map_40=class_map_40, class_map_13=class_map_13)
    #generate_csv_dataset_train(input_file=file_name, train_idx=train_idx, class_map_13=class_map_to_13)
    #print('Generated train dataset success.')
    #generate_csv_dataset_test(input_file=file_name, train_idx=train_idx, test_idx=test_idx, class_map_40=class_map_40, class_map_13=class_map_13)
    generate_csv_dataset_test(input_file=file_name, test_idx=test_idx, class_map_13=class_map_to_13)
    print('Generated test dataset success.')
    #generate_csv_dataset(input_file=file_name, train_idx=train_idx, test_idx=test_idx, class_map_40=class_map_40, class_map_13=class_map_13)
