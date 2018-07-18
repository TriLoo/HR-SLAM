# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.14'

from mxnet import nd
from mxnet import gluon
from mxnet import image
#import scipy.io as scio    # please use HDF reader for matlab v7.3 files
import h5py

'''
Dataset:  
    NYU Dataset V2
Method:
    Input: .mat file
    Output: Data set of mxnet
    Tool: scipy.io
'''

mat_dir = '/home/smher/Documents/DL_Datasets/NYUv2/nyu_depth_v2_labeled.mat'

# TODO: calculation following four numbers
rgb_mean = nd.array([112, 112, 112])
depth_mean = nd.array([112])
rgb_std = nd.array([1, 1, 1])
depth_std = nd.array([1])

def rand_crop(data, label, shape):
    data, rect = image.random_crop(data, shape)
    label = image.fixed_crop(label, *rect)

    return data, label


augs = gluon.data.vision.transforms.Compose(
    [
        gluon.data.vision.transforms.RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),     # input tensor with (H, W, C)
    ]
)


def normalize_img(rgb, depth):
    data_rgb = (rgb.astype('float32')/255.0 - rgb_mean) / rgb_std
    data_depth = (depth.astype('float32')/255.0 - depth_mean) / depth_std

    return data_rgb, data_depth


if __name__ == '__main__':
    f = h5py.File(mat_dir)
    print(list(f))   # refs, subsymtem, accelData, depths, iamges, instances, labels, names, namesToIds, rawDepthFilenames, rawDeths ...
    print(f['depths'].shape)     # (1449, 640, 480)
    print(f['images'].shape)    # (1449, 3, 640, 480)