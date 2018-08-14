# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.20'

import mxnet as mx
from mxnet import nd
import numpy as np
import readMat
import ResNet3D
import os

import joblib


def calculate_mean_std(img):
    mean_val = np.mean(img, axis=(0, 1))
    std_val = np.std(img, axis=(0, 1))
    return mean_val, std_val


def normalize_img(img, hyperspectral_mean=0, hyperspectral_std=1):
    return (img.astype('float32') - hyperspectral_mean) / hyperspectral_std   # 输入的图像已经是在


def detection_softmax(testmat='Indian_pines_corrected.mat',
                 gtmat='Indian_pines_gt.mat',
                 num_classes=9, net_params='ResNet3D.params',
                 ctx=mx.cpu(),
                 win_width=9, win_height=9,
                 mean_file='IndianPines_mean.joblib',
                 std_file='IndianPines_std.joblib',
                 results_file='result_file.joblib',
                 is_save=True):

    # Model Preparation
    net = ResNet3D.ResNet3D(num_classes)
    if not os.path.exists(net_params):
        raise NameError('No parameter exits.')
    net.collect_params().load(net_params, ctx=ctx)
    net.hybridize()

    # Data Preparation
    raw_data = readMat.readMatFile(testmat)
    splited_data = readMat.generate_child_window(raw_data, height=win_height, width=win_width)
    rows = splited_data['Rows'].astype('int')
    cols = splited_data['Cols'].astype('int')
    test_data_list = splited_data['Datas']
    hyper_mean, hyper_std = calculate_mean_std(raw_data)

    pred_lst = []

    # Start prediction.
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            data = test_data_list[idx]
            h, w, d = data.shape
            data = normalize_img(data, hyper_mean, hyper_std).reshape((1, 1, h, w, d))
            data = nd.array(data).transpose((0, 1, 4, 2, 3))
            data = data.copyto(ctx)
            pred = net(data)
            pred = pred.copyto(mx.cpu())
            pred_lst.append(pred)

    if is_save:
        # Save results.
        try:
            joblib.dump(pred_lst, results_file)
            print('Have done {} windows.'.format(len(pred_lst)))
            print('Predictions have been saved.')
        except:
            print('Predictions save failed.')

    return pred_lst


if __name__ == '__main__':
    detection_softmax()
