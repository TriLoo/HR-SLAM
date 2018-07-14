# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.11'

import argparse

import mxnet as mx
from mxnet import gluon
import joblib

import ResNet


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dir', default='/home/slz/.mxnet/datasets/CUB2011_test')
    parse.add_argument('--params', default='./resnet.params', help='The pretrained params file for resnet')
    parse.add_argument('--class_nums', type=int, default=100, help='The number of classes for testing')
    parse.add_argument('--scores', default='scores.joblib')
    parse.add_argument('--labels', default='labels.joblib')
    parse.add_argument('--context', default=mx.gpu(), help='The context where the data and operations running: mx.cpu(), mx.gpu()')
    args = parse.parse_args()

    test_set = gluon.data.vision.ImageFolderDataset(args.dir)
    test_data = gluon.data.DataLoader(test_set)

    net = ResNet.finetune_resnet18
    net.collect_params().load(args.params)
    net.collect_params().reset(args.context)

    scores = [[] for _ in range(args.class_nums)]
    labels = [[] for _ in range(args.class_nums)]

    for i, batch in enumerate(test_data):
        data, label = batch
        data = data.copyto(args.context)
        label = label.copyto(args.context)
        pred = net(data)

        labels.append(label.ascalar())
        scores.append(pred.asnumpy())

    joblib.dump(scores, args.scores)
    joblib.dump(labels, args.labels)

    print('Test data results (scores, labels) saved.')


if __name__ == '__main__':
    main()
