# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.19'

import scipy.io as scio
import numpy as np


def readMatFile(filename = 'Indian_pines_corrected.mat'):
    data = scio.loadmat(filename)
    print(list(data))
    indian_data = data['indian_pines_corrected']
    #indian_data = data['indian_pines_gt']
    print('type of indian data: ', type(indian_data))     # numpy.ndarray
    print('shape of indian data: ', indian_data.shape)    # (145, 145, 200)
    minval = np.min(indian_data)
    maxval = np.max(indian_data)
    indian_data = (indian_data - minval) / maxval
    return indian_data


def generate_child_window(data, width=9, height=9):
    data_shape = data.shape
    rows = np.floor(data_shape[0] / height)
    cols = np.floor(data_shape[1] / width)
    datas = []
    boxes = []
    row_idx = np.linspace(0, rows - 1, rows)
    col_idx = np.linspace(0, cols - 1, cols)
    for i in row_idx:           # row second
        for j in col_idx:       # column first
            ele = data[(i * height).astype('int'): ((i + 1) * height).astype('int'), (j * width).astype('int') : ((j + 1) * width).astype('int')]
            datas.append(ele)
            center_x = i * width + (width >> 1)
            center_y = j * height + (height >> 1)
            boxes.append([(center_x, center_y), width, height])

    return {'Datas': datas, 'Rows': rows, 'Cols': cols, 'Boxes': boxes}


# Indian_pines_corrected : indian_pines_corrected
# Indian_pines_gt        : indian_pines_gt
if __name__ == '__main__':
    data = readMatFile('Indian_pines_corrected.mat')
    #minval = np.min(data)
    #maxval = np.max(data)
    #data = (data - minval) / maxval
    #data = readMatFile('Indian_pines_gt.mat')
    print('min of data: ', np.min(data))      # label: 0    # old data: 955     # new data:  0.0
    print('max of data: ', np.max(data))      # label: 16   # old data: 9604    # new data:  0.9
    #print(data)
    #plt.imshow(data, cmap='gray')
    #plt.show()
    split_data = generate_child_window(data)
    print(split_data['Rows'], split_data['Cols'])     # 16, 16
    print('length of split_data: ', len(split_data['Windows']))    # 256
    print('size of windows: ', split_data['Windows'][0].shape)     # (9, 9, 200)

