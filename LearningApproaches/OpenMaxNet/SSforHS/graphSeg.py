# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.09.12'

# "Efficient Graph-Based Image Segmentation - Pedro F. Felzenszwalb etc."


from filter import *
from segment_graph import *
import time
import numpy as np
import cv2
import readRaw


def segment_hs(hs_img, sigma, k, min_size):
    start_time = time.time()
    bands, height, width = hs_img.shape
    hs_img = np.transpose(hs_img, axes=(1, 2, 0)) # change the bands to the last dims

    smooth_bands = []
    for idx in range(bands):
        img = hs_img[:, :, idx]
        smooth_img = smooth(img, sigma)
        smooth_bands.append(smooth_img)
    smooth_bands = np.array(smooth_bands)
    smooth_bands = np.transpose(smooth_bands, axes=(1, 2, 0))
    print('shape of smooth_bands: ', smooth_bands.shape)

    # build graph
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x+1))
                edges[num, 2] = np.sqrt(np.dot(smooth_bands[y, x, :], smooth_bands[y, x + 1, :]))
                num += 1
            if y < height - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y+1) * width + x)
                edges[num, 2] = np.sqrt(np.dot(smooth_bands[y, x, :], smooth_bands[y + 1, x, :]))
                num += 1
            if (x < width - 1) and (y < height - 2):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y+1) * width + (x+1))
                edges[num, 2] = np.sqrt(np.dot(smooth_bands[y, x, :], smooth_bands[y + 1, x + 1, :]))
                num += 1
            if (x < width - 1) and (y > 0):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y-1) * width + (x+1))
                edges[num, 2] = np.sqrt(np.dot(smooth_bands[y, x, :], smooth_bands[y - 1, x + 1, :]))
                num += 1

    print('graph construct done.')
    u = segment_graph(width * height, num, edges, k)

    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    #output = np.zeros(shape=(height, width, 3))
    output = np.zeros(shape=(height, width))

    # pick random colors for each component
    colors = np.zeros(shape=(height * width, 3))
    #colors = np.zeros(shape=(height * width))
    for i in range(height * width):
        colors[i, :] = random_rgb()
        #colors[i] = i

    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            output[y, x] = np.mean(colors[comp])
            #output[y, x, :] = colors[comp, :]

    elapsed_time = time.time() - start_time
    print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
             int(elapsed_time % 60)) + " seconds")

    #print('shape of result: ', output.shape)
    #print('type of result: ', type(output))
    #print('max value of result: ', np.max(output))
    #print('min value of result: ', np.min(output))
    output = output.astype(np.uint8)

    #cv2.imshow('Test Output', output)
    #cv2.waitKey()

    return output


if __name__ == '__main__':
    sigma = 0.5
    k = 500
    min_size = 50
    hs_img = readRaw.read_raw_data()
    print('hs_img shape: ', hs_img.shape)   # (128, 696, 587)
    segment_hs(hs_img, sigma, k, min_size)




