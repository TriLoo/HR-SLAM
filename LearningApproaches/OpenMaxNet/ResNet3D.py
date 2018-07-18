# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.18'

import mxnet as mx
from mxnet import nd
from mxnet import gluon


def get_conv3d(out_channels, ks, stride, padding, act=True):
    net = gluon.nn.HybridSequential()
    net.add(
        gluon.nn.Conv3D(out_channels, kernel_size=ks, strides=stride, padding=padding),
        gluon.nn.BatchNorm()
    )
    if act:
        net.add(gluon.nn.Activation('relu'))

    return net


class ResNet3D_Downsample(gluon.nn.HybridBlock):
    def __init__(self, out_channels, ks, stride,  padding, **kwargs):
        super(ResNet3D_Downsample, self).__init__(**kwargs)
        mid_channels = out_channels >> 1
        with self.name_scope():
            self.conv3_1 = get_conv3d(out_channels=mid_channels, ks=ks, stride=1, padding=padding, act=True)
            self.conv3_2 = get_conv3d(out_channels=out_channels, ks=ks, stride=stride, padding=padding, act=False)
            self.conv3_side = get_conv3d(out_channels=out_channels, ks=1, stride=stride, padding=0, act=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        conv3_1 = self.conv3_2(self.conv3_1(x))
        conv3_1_side = self.conv3_side(x)
        return F.relu(conv3_1 + conv3_1_side)


class ResNet3D_IdentityLayer(gluon.nn.HybridBlock):
    def __init__(self, out_channels, **kwargs):
        super(ResNet3D_IdentityLayer, self).__init__(**kwargs)
        mid_channels = out_channels >> 1
        with self.name_scope():
            self.conv3_1 = get_conv3d(out_channels=out_channels, ks=1, stride=1, padding=0, act=True)
            self.conv3_2 = get_conv3d(out_channels=mid_channels, ks=3, stride=1, padding=1, act=True)
            self.conv3_3 = get_conv3d(out_channels=out_channels, ks=1, stride=1, padding=0, act=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        conv3 = self.conv3_3(self.conv3_2(self.conv3_1(x)))
        return F.relu(conv3 + x)


class ResNet3D(gluon.nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(ResNet3D, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv3D(4, kernel_size=3, strides=1, padding=1, activation='relu')   # output shape (1, 4, 200, 9, 9 )
            self.layer1 = ResNet3D_Downsample(16, (4, 3, 3), (4, 2, 2), 1)         # output shape (1, 16, 50, 5, 5)
            self.layer2 = ResNet3D_IdentityLayer(16)                 # output shape (1, 16, 50, 5, 5)
            self.layer3 = ResNet3D_Downsample(32, 3, (3, 2, 2), 1)         # output shape (1, 32, 24, 3, 3)
            self.layer4 = ResNet3D_IdentityLayer(32)                 # output shape (1, 32, 12, 3, 3)
            self.flatten = gluon.nn.Flatten()                              # output shape (1, 32 * 12 * 3 * 3 = 3456)
            self.denselayer1 = gluon.nn.Dense(2048, activation='relu')
            self.dropout1 = gluon.nn.Dropout(0.4)
            self.denselayer2 = gluon.nn.Dense(1024, activation='relu')
            self.dropout1 = gluon.nn.Dropout(0.4)
            self.denselayer2 = gluon.nn.Dense(512, activation='relu')
            self.outputlayer = gluon.nn.Dense(num_classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        layer1 = self.layer1(self.conv1(x))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        flattened = self.flatten(layer4)
        dropout1 = self.dropout1(self.denselayer1(flattened))
        dense2 = self.denselayer2(dropout1)
        outputs = self.outputlayer(dense2)
        return outputs


# for test the model
if __name__ == '__main__':
    net = ResNet3D(10)
    net.initialize(mx.init.Xavier())
    net.hybridize()
    x = nd.random_uniform(0,1, shape=(1, 1, 200, 9, 9))   # the shape is same as training samples
    y = net(x)
    print(y.shape)   # output (1, 10)


'''
net = gluon.nn.Sequential()
net.add(gluon.nn.Conv3D(16, (3, 3, 3), padding=(1, 1, 1), activation='relu'))    # layout: NCDHW, weight's shape: (out_channels, input_channels, 3, 3, 3)
net.initialize()

#x = nd.random_uniform(0, 1, shape=(1, 1, 200, 9, 9))    # output: (1, 16, 200, 9, 9)
x = nd.random_uniform(0, 1, shape=(1, 1, 200, 9, 9))    # output: (1, 16, 200, 9, 9)

y = net(x)
print(net.collect_params())
print(y.shape)
'''
