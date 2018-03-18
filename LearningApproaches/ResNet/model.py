import mxnet as mx
from mxnet import gluon
from mxnet import nd
import numpy as np
from mxnet import autograd

'''
    This file includes the model of ResNet-50
    Using bootleneck blocking.
'''

# Bottleneck Architectures
class Residual(gluon.nn.Block):
    def __init__(self, channelA, channelB, sameshape = True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.shapeSame = sameshape
        with self.name_scope():
            self.net = gluon.nn.Sequential()
            self.net.add(
                gluon.nn.Conv2D(channels=channelA, kernel_size=(1, 1), strides=2, activation='relu'),
                gluon.nn.BatchNorm(),
                gluon.nn.Conv2D(channels=channelA, kernel_size=(3, 3), strides=2, activation='relu'),
                gluon.nn.BatchNorm(),
                gluon.nn.Conv2D(channels=channelB, kernel_size=(1, 1), strides=2),
                gluon.nn.BatchNorm()
            )
            if not sameshape:
                self.projection = gluon.nn.Conv2D(channelA, kernel_size=(1, 1), strides=2)


    def forward(self, x):
        output = self.net(x)
        if not self.shapeSame:
            x = self.projection(x)
        return x + output

class ResNet(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        with self.name_scope():
            # conv1: 7 * 7, 64, stride 2
            a1 = gluon.nn.Sequential()
            a1.add(
                gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), strides=(2, 2), padding=0)
            )

            # conv2:
            a2 = gluon.nn.Sequential()
            a2.add(
                Residual(channelA=64, channelB=256),
                Residual(channelA=64, channelB=256),
                Residual(channelA=64, channelB=256),
            )

            # conv3:
            a3 = gluon.nn.Sequential()
            a3.add(
                Residual(channelA=128, channelB=512),
                Residual(channelA=128, channelB=512),
                Residual(channelA=128, channelB=512),
                Residual(channelA=128, channelB=512),
            )

            # conv4:
            a4 = gluon.nn.Sequential()
            a4.add(
                Residual(channelA=256, channelB=1024),
                Residual(channelA=256, channelB=1024),
                Residual(channelA=256, channelB=1024),
                Residual(channelA=256, channelB=1024),
                Residual(channelA=256, channelB=1024),
                Residual(channelA=256, channelB=1024),
            )

            # conv5:
            a5 = gluon.nn.Sequential()
            a5.add(
                Residual(channelA=512, channelB=1024),
                Residual(channelA=512, channelB=1024),
                Residual(channelA=512, channelB=1024),
            )

            self.net = gluon.nn.Sequential()
            self.net.add(a1, a2, a3, a4, a5,
                         gluon.nn.AvgPool2D(pool_size=(2, 2))
                         )

    def forward(self, x):
        output = nd.softmax(self.net(x))
        return output



# Define the training function
def train(net, train_data, test_data, batchsize, epochs):
    totalloss = [np.mean(net(train_data).asnumpy())]

    #for epoch in range(1, epochs):
