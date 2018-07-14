# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.14'

import h5py

def readH5(filename):
    f = h5py.File(filename)
    print('Keys of h5 file: ', f.keys())
    datas = f['data'][:]   # return np.ndarray
    labels = f['label'][:]
    f.close()

    return datas, labels

'''
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 120

f = h5py.File('I9-9.h5')    # dataset: Indian Pines, spectral * 200.

print(list(f))      # return ['data', 'label']

print(type(f))          # return class 'h5py._h1.files.File'
print(type(f['data']))  # return class 'h5py._h1.dataset.Dataset'
print(f['data'].shape)  # return (9234, 16200), Note: ! ! ! 16200 = 9 * 9 * 200 ! ! !
print(f['label'].shape)  # return (9324, 9)

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
