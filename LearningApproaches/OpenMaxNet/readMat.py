# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.19'

import scipy.io as scio
import numpy as np


def readMatFile(filename = 'Indian_pines_corrected.mat'):
    data = scio.loadmat(filename)
    indian_data = data['indian_pines_corrected']
    print('type of indian data: ', type(indian_data))     # numpy.ndarray
    print('shape of indian data: ', indian_data.shape)    # (145, 145, 200)
    return indian_data


def generate_child_window(data, height = 9, width = 9):
    data_shape = data.shape
    rows = np.floor(data_shape[0] / height)
    cols = np.floor(data_shape[1] / width)
    windows = []
    row_idx = np.linspace(0, rows - 1, rows)
    col_idx = np.linspace(0, cols - 1, cols)
    for i in row_idx:           # row second
        for j in col_idx:       # column first
            ele = data[(i * height).astype('int'): ((i + 1) * height).astype('int'), (j * width).astype('int') : ((j + 1) * width).astype('int')]
            windows.append(ele)

    return {'Rows': rows, 'Cols': cols, 'Windows': windows}


if __name__ == '__main__':
    data = readMatFile()
    split_data = generate_child_window(data)
    print(split_data['Rows'], split_data['Cols'])     # 16, 16
    print('size of windows: ', len(split_data['Windows']))    # 256

