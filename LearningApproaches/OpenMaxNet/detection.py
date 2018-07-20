# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.20'

import mxnet as mx
from mxnet import nd
import numpy as np
import readMat
import ResNet3D

import argparse
import joblib


def calculate_mean_std(img):
    mean_val = np.mean(img, axis=(0, 1))
    std_val = np.std(img, axis=(0, 1))
    return mean_val, std_val


def normalize_img(img, hyperspectral_mean, hyperspectral_std):
    return (img.astype('float32') - 0) / 1   # 输入的图像已经是在
#return (img.astype('float32') - hyperspectral_mean) / hyperspectral_std   # 输入的图像已经是在


def detection(is_save=True):
    parse = argparse.ArgumentParser()
    parse.add_argument('--testmat', default='Indian_pines_corrected.mat', help='The file under test.')
    parse.add_argument('--gtmat', default='Indian_pines_gt.mat', help='The ground truth file.')
    parse.add_argument('--num_classes', type=int, default=9, help='The known classes in training datasets.')
    parse.add_argument('--net_params', default='ResNet3D.params', help='The parameter file of pretrained networks.')
    parse.add_argument('--ctx', type=mx.Context, default=mx.gpu(), help='The running context of all datas.')
    parse.add_argument('--win_height', type=int, default=9, help='Set the height of each windows.')
    parse.add_argument('--win_width', type=int, default=9, help='Set the width of each windows.')
    parse.add_argument('--mean_file', default='IndianPines_mean.joblib', help='Used for data preprocessing.')
    parse.add_argument('--std_file', default='IndianPines_std.joblib', help='Used for data preprocessing.')
    parse.add_argument('--result_file', default='result_file.joblib', help='The file name results are written to.')
    args = parse.parse_args()

    # Model Preparation
    net = ResNet3D.ResNet3D(args.num_classes)
    net.collect_params().load(args.net_params, ctx=args.ctx)
    net.hybridize()

    # Data Preparation
    raw_data = readMat.readMatFile(args.testmat)
    splited_data = readMat.generate_child_window(raw_data, height=args.win_height, width=args.win_width)
    rows = splited_data['Rows'].astype('int')
    cols = splited_data['Cols'].astype('int')
    test_data_list = splited_data['Windows']
    hyper_mean, hyper_std = calculate_mean_std(raw_data)
#hyper_mean = joblib.load(args.mean_file)
#hyper_std = joblib.load(args.std_file)

    # To store the result.
    pred_lst = []

    # Start prediction.
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            data = test_data_list[idx]
            h, w, d = data.shape
            data = normalize_img(data, hyper_mean, hyper_std).reshape((1, 1, h, w, d))
#print('min value of data: ', np.min(data))
#print('max value of data: ', np.max(data))
            data = nd.array(data).transpose((0, 1, 4, 2, 3))
            data = data.copyto(args.ctx)
            pred = net(data)
            pred = pred.copyto(mx.cpu())
            pred_lst.append(pred)
#print('predicted class: ', np.argmax(pred.asnumpy()))
            #print('type of input data: ', type(data))
            #print('shape of input data: ', data.shape)
            #print('type of pred: ', type(pred))
            #print('shape of pred: ', pred.shape)
            #print('length of pred: ', len(pred_lst))

    if is_save:
        # Save results.
        try:
            joblib.dump(pred_lst, args.result_file)
            print('Have done {} windows.'.format(len(pred_lst)))
            print('Predictions have been saved.')
        except:
            print('Predictions save failed.')
    else:
        return pred_lst


if __name__ == '__main__':
    detection()
