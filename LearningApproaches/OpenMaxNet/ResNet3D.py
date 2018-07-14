# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.14'

import mxnet as mx
from mxnet import nd
from mxnet import gluon

net = gluon.nn.Sequential()
net.add(gluon.nn.Conv3D(16, (3, 3, 3), padding=(1, 1, 1), activation='relu'))    # layout: NCDHW, weight's shape: (out_channels, input_channels, 3, 3, 3)
net.initialize()


#x = nd.random_uniform(0, 1, shape=(1, 1, 200, 9, 9))    # output: (1, 16, 200, 9, 9)
x = nd.random_uniform(0, 1, shape=(1, 1, 200, 9, 9))    # output: (1, 16, 200, 9, 9)

y = net(x)
print(net.collect_params())
print(y.shape)
