# -*- coding: utf-8 -*-

##
#@date 2018.08.14
#@brief 修改，将detection_om, detection_sm整合到此文件
#

__author__ = 'smh'
__date__ = '2018.08.14'

import mxnet as mx
from mxnet import nd
import numpy as np
import readMat
import ResNet3D
import os

import joblib
from OpenMaxLayer import fit_weibull, openmax


def calculate_mean_std(img):
    mean_val = np.mean(img, axis=(0, 1))
    std_val = np.std(img, axis=(0, 1))
    return mean_val, std_val


def normalize_img(img, hyperspectral_mean=0, hyperspectral_std=1):
    return (img.astype('float32') - hyperspectral_mean) / hyperspectral_std   # 输入的图像已经是在


def __detection_softmax(net, rows, cols, test_data_list, hyper_mean, hyper_std, ctx,
                      results_file='softmax_action_vector.joblib',
                      is_save=True):
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

    print('SoftMax Calculation Finished.')
    return pred_lst


def __detection_openmax(mavs, dists, pred_lst, class_num=9,
                      alpha=3, tailsize=20,
                      distances_type='eucos',
                      euc_scale=5e-3,
                      threshold=0.0,
                      is_save=True):
    categories = [i for i in range(class_num)]
    # 得到每一类的weibull model，
    weibull_model = fit_weibull(mavs, dists, categories, tailsize, distances_type)

    pred_sm, pred_om = [], []

    for i, score in enumerate(pred_lst):    # for temp test
        score = score.asnumpy()
        so, ss = openmax(weibull_model, categories, score, euc_scale, alpha, distances_type)
        pred_sm.append(np.argmax(ss) if np.max(ss) >= threshold else 9)
        pred_om.append(np.argmax(so) if np.max(so) >= threshold else 9)

    if is_save:
        joblib.dump(pred_sm, 'detection_softmax.joblib')
        joblib.dump(pred_om, 'detection_openmax.joblib')

    print('OpenMax Calculation Finished.')
    return pred_om, pred_sm


# 返回：openmax, softmax, boxes
def detection_om_sm(num_classes, net_params, ctx, splited_data,
                    mavs, dists, is_save=True
                    ):
    # Model Preparation
    net = ResNet3D.ResNet3D(num_classes)
    if not os.path.exists(net_params):
        raise NameError('No parameter exits.')
    net.collect_params().load(net_params, ctx=ctx)
    net.hybridize()

    # Data Preparation
    rows = splited_data['Rows'].astype('int')
    cols = splited_data['Cols'].astype('int')
    test_data_list = splited_data['Datas']
    #hyper_mean, hyper_std = calculate_mean_std(raw_data)
    hyper_mean, hyper_std = (0, 1)

    # softmax activation vectors
    pred_lst = __detection_softmax(net, rows, cols, test_data_list, hyper_mean, hyper_std, ctx=mx.cpu(), is_save=is_save)

    # openmax prediction
    pred_om, pred_sm = __detection_openmax(mavs, dists, pred_lst, is_save=is_save)

    return (pred_om, pred_sm)


if __name__ == '__main__':
    testmat = 'Indian_pines_corrected.mat'
    win_height, win_width = (9, 9)
    raw_data = readMat.readMatFile(testmat)
    mavs = joblib.load('mavs.joblib')
    dists = joblib.load('dists.joblib')
    splited_data = readMat.generate_child_window(raw_data, height=win_height, width=win_width)
    om, sm = detection_om_sm(9, 'ResNet3D.params', mx.cpu(), splited_data, mavs, dists)

