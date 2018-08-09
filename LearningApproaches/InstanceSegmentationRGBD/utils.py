# -*- coding:utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.09'

import os
import numpy as np
from scipy.io import loadmat
import h5py
import joblib
import argparse

nyu_dir = '/home/smher/Documents/DL_Datasets/NYUv2/nyu_depth_v2_labeled.mat'


def __calculate_means(img, mean_files):
    ''' 分成十次计算 '''
    img_len = img.shape[0]
    curr_mean = 0
    if img_len >= 1000:
        step = img_len / 10
        for i in range(10):
            start = step * i
            end = step * (i + 1)
            curr_imgs = img[np.int32(start):np.int32(end), :, :, :]
            curr_mean += np.mean(curr_imgs, axis=(0, 2, 3))
        curr_mean /= 10
    else:
        curr_mean = np.mean(img, axis=(0, 2, 3))
    joblib.dump(curr_mean, mean_files)
    print('Mean datas has been saved {}.'.format(mean_files))


def __calculate_stds(img, std_files):
    '''
      同样分成十次计算，避免存储爆掉

    :param img:
    :param std_files:
    :return:
    '''

    img_len = img.shape[0]
    curr_std = 0
    total_std = 0
    if img_len >= 1000:
        step = img_len / 10
        for i in range(10):
            start = step * i
            end = step * (i + 1)
            curr_imgs = img[np.int32(start):np.int32(end), :, :, :]
            curr_std = np.std(curr_imgs, axis=(0, 2, 3))
            total_std += curr_std * curr_std
        total_std = np.sqrt(total_std / 10)
    else:
        total_std = np.std(img, axis=(0, 2, 3))
    joblib.dump(total_std, std_files)
    print('Std datas has been saved to {}.'.format(std_files))


def calculate_mean_std():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_file', default=nyu_dir)
    parse.add_argument('--mean_img_file', default='nyu_image_means.joblib')
    parse.add_argument('--std_img_file', default='nyu_image_stds.joblib')
    parse.add_argument('--mean_dep_file', default='nyu_depth_means.joblib')
    parse.add_argument('--std_dep_file', default='nyu_depth_stds.joblib')
    args = parse.parse_args()

    exts = os.path.splitext(args.data_file)[1]
    images = None
    depths = None
    if exts == '.mat':
        print('Reading matlab data file.')
        try:
            print('Reading mat file using loadmat.')
            f = loadmat(args.data_file)
        except:
            print('Reading mat file using h5py.')
            f = h5py.File(args.data_file)

    images = f['images'][:]
    depths = f['depths'][:]
    f.close()
    if len(depths.shape) == 3:
        depths = depths[:, np.newaxis, :, :]

    if images is not None:
        print('shape of iamges: ', images.shape)
        __calculate_means(images, args.mean_img_file)
        __calculate_stds(images, args.std_img_file)
    else:
        print('No image data.')

    if depths is not None:
        print('shape of depths: ', depths.shape)
        __calculate_means(depths, args.mean_dep_file)
        __calculate_stds(depths, args.std_dep_file)
    else:
        print('No depth  data.')

    print('Done.')


if __name__ == '__main__':
    calculate_mean_std()
    a = joblib.load('nyu_depth_stds.joblib')
    print('std of depth image: ', a)
