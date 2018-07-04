import os
import argparse
import _pickle as cPickle
import joblib

import mxnet as mx
from mxnet import gluon
from mxnet import nd

import numpy as np
import DarkNet19
import readCUBData

# Total 100 classes

batch_size = 1
class_nums = 100

data_dir = '/home/slz/.mxnet/datasets/CUB100'


if __name__ == '__main__':
    train_set = gluon.data.vision.ImageFolderDataset(data_dir, transform=lambda X, y: readCUBData.transform(X, y, readCUBData.augs))
    train_data = gluon.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    scores = [[] for _ in range(class_nums)]

    net = DarkNet19.finetune_resnet18
#net.collect_params().load('resnet_back.params', ctx=mx.gpu())
    net.collect_params().load('resnet_back.params')
    net.collect_params().reset_ctx(mx.gpu())      # to reload the model, should use 'reset_ctx' of ParameterDict or Parameter class

    for i, batch in enumerate(train_data):
        data, label = batch
        data = data.copyto(mx.gpu())
        label = label.copyto(mx.gpu())
        pred = net(data)
        pred = pred.copyto(mx.cpu()).asnumpy()
#print('shape of preds = ', pred.shape) # (1, 100)
#print('shape of label = ', label.shape) # (1, 100)
        if np.argmax(pred, axis=1) == label.asnumpy():
            idx = label.asscalar().astype('uint8')
            scores[idx].append(pred)

    scores = [ np.array(x)[:, np.newaxis, :] for x in scores ]
    mavs = np.array([np.mean(x, axis=0) for x in scores])

    joblib.dump(scores, 'train_scores.joblib')
    joblib.dump(mavs, 'mavs.joblib')

    for i in range(100):
        print('%d-th scores length is %d'%(i, len(scores[i])))

    print('Parameter saved.')


