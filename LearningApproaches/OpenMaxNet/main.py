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
    parse.add_argument('--batch_size', type=int, default=2)
    parse.add_argument('--epoches', type=int, default=1)
    augs = parse.parse_args()

    ctx = mx.cpu()
    epoches = augs.epoches
    batch_size = augs.batch_size
    acc = mx.metric.Accuracy()
    dataset = readH5.IndianDatasets(augs.filename)
    train_data = gluon.data.DataLoader(dataset, batch_size=batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    net = ResNet3D.ResNet3D(9)     # the number of classes is 9 ...
    net.initialize(mx.init.Xavier(), ctx=ctx)
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
                loss_t = loss(nd.argmax(pred), nd.argmax(label))
            loss_t.backward()
            trainer.step(batch_size)
            #print('shape of (label): ', label.shape)                 # (2, 9)
            #print('shape of (pred): ', pred.shape)                   # (2, 9)
            #print('shape of argmax(label): ', nd.argmax(label, axis=1).shape) # (2, )
            #print('shape of argmax(pred): ', nd.argmax(pred, axis=1).shape)   # (2, )
            #acc.update([nd.argmax(label)], [nd.argmax(pred)])

            if ((i+1) % 100) == 0:
                print('shape of lost_t: ', loss_t.shape)
                print('type of loss_t', type(loss_t))
                print('current loss = ', loss_t.asnumpy())

        if epoch == 30:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        if epoch == 60:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        if epoch == 80:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)

    #print('Epoch: %d, training acc: %s %d'%(epoch, *acc.get()))


if __name__ == '__main__':
    main()


