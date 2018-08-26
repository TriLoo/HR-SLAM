# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.22'

import numpy as np
import model
import cv2

dir = '/home/smher/Documents/Hyperspectrals/20180822/hs_img_datas_0001.raw'


def read_raw_data(file_name, samples, lines, bands):
    fd = open(file_name, 'rb')
    rows = lines
    cols =  samples * bands
    bin_img = np.fromfile(fd, dtype=np.uint16, count=rows * cols)
    fd.close()
    bin_img = np.reshape(bin_img, newshape=(rows, bands, -1))
    img = np.transpose(bin_img, axes=(1, 0, 2))
    return img


if __name__ == '__main__':
    rgb = read_raw_data(dir, samples=696, lines=587, bands=128)
    rgb = np.float32(rgb)
    img = model.show_hyper_img_top(rgb)
    img = img.astype('float32')
    img = img / np.max(img)
    cv2.imshow('Test', img)
    cv2.waitKey()
    '''
    fd = open(dir, 'rb')
    rows = 587
    cols = 696 * 128
    bin_img = np.fromfile(fd, dtype=np.uint16, count=rows * cols)
    fd.close()
    print('type of im: ', type(bin_img))    # numpy.ndarray
    print('date type of im: ', bin_img.dtype)
    print('shape of im: ', bin_img.shape)   # (rows * cols)
    bin_img = np.reshape(bin_img, newshape=(rows, 128, -1))
    bin_img = np.transpose(bin_img, axes=(0, 2, 1))
    r = bin_img[:, :, 55]
    g = bin_img[:, :, 85]
    b = bin_img[:, :, 93]
    rgb = np.array([b, g, r])
    rgb = rgb / np.max(rgb)
    rgb = np.transpose(rgb, axes=(2, 1, 0))
    cv2.imshow('Test', rgb)
    cv2.waitKey()
    '''
    '''
    img = open_image(dir)
    print (type(img))

    img = read_row_data()
    rgb_img = model.show_hyper_img_top(img, True)
    print('Test passed.')
    '''
