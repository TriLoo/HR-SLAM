import mxnet as mx
from mxnet import gluon
import numpy as np

class FlowNetS(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(FlowNetS, self).__init__(**kwargs)
        self.conv1 = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), strides=(2, 2), padding=1, activation='relu')
        self.conv2 = gluon.nn.Conv2D(channels=128, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu')
        self.conv3 = gluon.nn.Conv2D(channels=256, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu')
        self.conv4 = gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv5 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(2, 2), padding=1, activation='relu')
        self.conv6 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv7 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=2, padding=1, activation='relu')
        self.conv8 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv9 = gluon.nn.Conv2D(channels=1024, kernel_size=(3 ,3), strides=1, padding=1, activation='relu')

        self.net = gluon.nn.Sequential()
        self.net.add(self.conv1, self.conv2)

    def forward(self, x):
        return self.net(x)
