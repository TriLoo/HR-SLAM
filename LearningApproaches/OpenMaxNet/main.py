# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.18'

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import nd

import ResNet3D
import readH5

import argparse


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--filename', default='I9-9.h5')
    parse.add_argument('--is_train', type=bool, default=True)
    parse.add_argument('--is_mavs', type=bool, default=False)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--epoches', type=int, default=12)
    augs = parse.parse_args()

    ctx = mx.gpu()
    epoches = augs.epoches
    batch_size = augs.batch_size
    acc = mx.metric.Accuracy()
    dataset = readH5.IndianDatasets(augs.filename)
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, last_batch='discard')
    #loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)
    net = ResNet3D.ResNet3D(9)     # the number of classes is 9 ...
    net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1, 'wd':0.0001})

    acc.reset()
    for epoch in range(epoches):
        for i, batch in enumerate(train_data):
            data, label = batch
            data = data.copyto(ctx)
            label = label.copyto(ctx)
            with autograd.record():
                pred = net(data)
                loss_t = loss(pred, nd.argmax(label, axis=1, keepdims=True))        # shape: batch_size * 1
            loss_t.backward()
            trainer.step(batch_size)
            acc.update(label.argmax(axis=1, keepdims=True), pred.argmax(axis=1, keepdims=True))

            if ((i+1) % 100) == 0:
                print('current loss = ', nd.sum(loss_t).asscalar())

        if epoch == 8:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        if epoch == 10:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        
        print('Epoch: %d, training acc: %s %.3f'%(epoch, *acc.get()))

    try:
        net.collect_params().save('ResNet3D.params')
    except:
        print('model params save failed.')


if __name__ == '__main__':
    main()


