# -*- coding: utf-8 -*-

'''
This file includes the calculation of mean activation vectors(MAV) with only training data!

input: activation vectors from network outputs
return three types distance forming a dict
'''

__author__ = 'smh'
__version__ ='0.1'
__date__ = '2018.07.10'

import argparse
import joblib

import mxnet as mx
from mxnet import gluon

import numpy as np
import ResNet
import readCUBData


batch_size = 1
class_nums = 100

data_dir = '/home/slz/.mxnet/datasets/CUB100'

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mavs_path', default='mavs.joblib')
    parse.add_argument('--scores_path', default='scores.joblib')
    args = parse.parse_args()

    train_set = gluon.data.vision.ImageFolderDataset(data_dir, transform=lambda X, y: readCUBData.transform(X, y, readCUBData.augs))
    train_data = gluon.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)

    scores = [[] for _ in range(class_nums)]    # 用于保存每一类别中分类正确的样本的Activation Vector, 共class_num个元素，每个元素对应一个list

    net = ResNet.finetune_resnet18
    net.collect_params().load('resnet.params')
    net.collect_params().reset_ctx(mx.gpu())     # 默认采用GPU

    for i, batch in enumerate(train_data):
        data, label = batch
        data = data.copyto(mx.gpu())      # shape = (1, class_nums)
        label = label.copyto(mx.gpu())    # shape(1, )

        pred = net(data)
        pred = pred.copyto(mx.cpu()).asnumpy()

        if np.argmax(pred, axis=1) == label.asnumpy():
            idx = label.asscalar().astype('uint8')
            scores[idx].append(pred)          # scores' shape: (class_num, N_c, 1, class_num)

    scores = [np.array(x) for x in scores]
    mavs = np.array([np.mean(x, axis=0) for x in scores])   # mavs' shape: (class_num, 1, class_num)

    #print('shape of scores elements: ', scores[0].shape)
    #print('shape of mavs: ', mavs.shape)

    joblib.dump(scores, args.scores_path)     # scores' shape = (class_num, N_c, 1, class_num), verified ! ! !
    joblib.dump(mavs, args.mavs_path)         # mavs' shape = (class_num, 1, class_num), verified ! ! !

    print('Parameters(scores, mavs) Saved.')


if __name__ == '__main__':
    main()
