# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.29'


import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ssModule
import cv2
import numpy as np


def ss_test():
    '''
    img = skimage.data.astronaut()
    print('type of img: ', type(img))
    print('shape of img: ', img.shape)
    print('data type of img: ', img.dtype)
    img = img.astype('float32')
    img /= np.max(img)
    '''
    #img = cv2.imread('hyperspectral_1bt.png')
    img = cv2.imread('hs_img_0001.bmp')
    img_lbl, regions = ssModule.selective_search(img, scale=500, sigma=0.9, min_size=10)

    candidatas = set()
    for r in regions:
        if r['rect'] in candidatas:
            continue
        if r['size'] < 2000:
            continue
        x, y, w, h = r['rect']
        if w/h > 1.2 or h/w > 1.2:
            continue
        candidatas.add(r['rect'])

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidatas:
        print(x, y, w, h)
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red',
                                  linewidth=1)
        ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    ss_test()


