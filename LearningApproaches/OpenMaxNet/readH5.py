# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.18'

import h5py
from mxnet import gluon
from mxnet import nd
import argparse
import numpy as np
import joblib


def readH5(filename):
    f = h5py.File(filename)
    print('Keys of h5 file: ', f.keys())
    datas = f['data'][:]   # return np.ndarray
    labels = f['label'][:]
    f.close()

    return datas, labels


# calculate channel-wise mean & std
def calculate_mean_std(filename):
    # data's shape: (9234, 16200), label's shape: (9234, 9), the label is one-hot coding style
    datas, labels = readH5(filename)
    print('datas shape: ', datas.shape)
    print('labels shape: ', labels.shape)
    datas_shape = datas.shape
    datas_newshape = np.reshape(datas, newshape=(datas_shape[0], 9, 9, -1))   # layout: NWHC = (9234, 9, 9, 200)
    data_mean = np.mean(datas_newshape, axis=(0, 1, 2))   # return the (200, ) array
    data_std = np.std(datas_newshape, axis=(0, 1, 2))
    return data_mean, data_std


hyperspectral_mean = joblib.load('IndianPines_mean.joblib')
hyperspectral_std = joblib.load('IndianPines_std.joblib')


# input img shape: (9, 9, 200)
def normalize_img(img):
    return (img.astype('float32') - hyperspectral_mean) / hyperspectral_std   # 输入的图像已经是在


class IndianDatasets(gluon.data.Dataset):
    def __init__(self, filename,  **kwargs):
        super(IndianDatasets, self).__init__(**kwargs)
        data, label = readH5(filename=filename)
        data = np.reshape(data, newshape=(9234, 9, 9, -1))  # layout: NWHC
        self.data = [normalize_img(img) for img in data]
        self.label = label
        print('Read ' + str(len(self.data)) + ' examples.')

    def __getitem__(self, item):
        data = self.data[item]      # numpy.NDarray
        label = self.label[item]    # numpy.NDarray
        data = np.reshape(data, newshape=(1, 9, 9, 200))
        data = nd.array(data).transpose((0, 3, 1, 2))
        label = nd.array(label)     # Convert the label from numpy.ndarray to nd.NDarray in mxnet! 上同
        return data, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--filename', default='I9-9.h5')
    augs = parse.parse_args()

    print('Testing the dataset class and data loader.')
    dataset = IndianDatasets(augs.filename)
    train_data = gluon.data.DataLoader(dataset, batch_size=1)
    for data, label in train_data:
        print(data.shape)    # return (1, 1, 200, 9, 9)  # layout: NCHW
        print(label.shape)   # return (1, 9)
        break
    print('Test Pass.')

    '''
    datas, labels = readH5(augs.filename)
    #print('min of datas: ', np.min(np.min(datas)))   # 0.099437
    #print('max of datas: ', np.max(np.max(datas)))   # 1.0
    data_mean, data_std = calculate_mean_std(augs.filename)
    print('mean = ', data_mean.shape)
    print('std = ', data_std.shape)

    joblib.dump(data_mean, 'IndianPines_mean.joblib')
    joblib.dump(data_std, 'IndianPines_std.joblib')
    '''


'''
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 120

f = h5py.File('I9-9.h5')    # dataset: Indian Pines, spectral * 200.

print(list(f))      # return ['data', 'label']

print(type(f))          # return class 'h5py._h1.files.File'
print(type(f['data']))  # return class 'h5py._h1.dataset.Dataset'
print(f['data'].shape)  # return (9234, 16200), Note: ! ! ! 16200 = 9 * 9 * 200 ! ! !
print(f['label'].shape)  # return (9234, 9)

datas = f['data'][:]
print(type(datas))  # return numpy.ndarray ! ! !
labels = f['label'][:]

f.close()

ele = datas[0]
labs = labels[0]

print(ele[10:20])
print('labels = ', labs)     # [0, 1, 0, 0, 0, 0, 0, 0, 0]: one-hot

print('shape of image: ', ele.shape)
ele = ele.reshape((9, 9, 200))
print(ele.shape)

img = ele[:, :, 0]

print(img)

print('shape of img: ', img.shape)
plt.imshow(img, cmap='gray')
plt.show()
'''
