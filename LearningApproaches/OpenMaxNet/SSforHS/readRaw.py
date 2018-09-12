# -*- coding: utf-8 -*-

'''
No need to read the file as 12bit. just use np.float16 as dtype
'''

__author__ = 'smh'
__date__ = '2018.08.22'

import cv2

dir = '/home/smher/Documents/Hyperspectrals/20180822/hs_img_datas_0001.raw'

import numba as nb
import numpy as np


@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)
def nb_read_uint12(data_chunk):
  """data_chunk is a contigous 1D array of uint8 data)
  eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)"""

  #ensure that the data_chunk has the right length
  assert np.mod(data_chunk.shape[0],3)==0

  out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)

  for i in nb.prange(data_chunk.shape[0]//3):
    fst_uint8=np.uint16(data_chunk[i*3])
    mid_uint8=np.uint16(data_chunk[i*3+1])
    lst_uint8=np.uint16(data_chunk[i*3+2])

    out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
    out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

  return out


def read_uint12(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8

    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])


def read_12bit_raw_data(file_name, samples, lines, bands):
    with open(file_name, 'rb') as fd:
        rows = lines
        cols = np.int(samples * bands * 2)
        bin_img = np.fromfile(fd, dtype=np.uint8, count=rows * cols)

    print('shape of bin_img: ', bin_img.shape)
    data = nb_read_uint12(bin_img)
    data = np.reshape(data, newshape=(lines, bands, -1))

    return np.transpose(data, axes=(1, 0, 2))


def read_raw_data(file_name=dir, samples=696, lines=587, bands=128):
    fd = open(file_name, 'rb')
    rows = lines
    cols = samples * bands
    bin_img = np.fromfile(fd, dtype=np.uint16, count=rows * cols)
    fd.close()
    bin_img = np.reshape(bin_img, newshape=(lines, bands, samples))
    img = np.transpose(bin_img, axes=(1, 2, 0))

    return img   # shape is: (128, 587, 696)


if __name__ == '__main__':
    samples = 696
    lines = 587
    bands = 128
    bin_img = read_raw_data(file_name=dir, samples=samples, lines=lines, bands=bands)
    r = bin_img[55, :, :]
    g = bin_img[85, :, :]
    b = bin_img[93, :, :]
    rgb = np.array([b, g, r])
    rgb = rgb / np.max(rgb)
    rgb = np.transpose(rgb, axes=(2, 1, 0))
    cv2.imshow('Test', rgb)
    cv2.waitKey()
