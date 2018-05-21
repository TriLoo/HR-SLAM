import mxnet as mx
from mxnet import gluon
import numpy as np

class FlowNetS(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(FlowNetS, self).__init__(**kwargs)
        self.conv1 = gluon.nn.Sequential()
        self.conv1.add(gluon.nn.Conv2D(channels=64, kernel_size=(7, 7)))
        self.conv2 = gluon.nn.Sequential()
        self.conv2.add(gluon.nn.Conv2D(channels=128, kernel_size=(5, 5)))

        self.net = gluon.nn.Sequential()
        self.net.add(self.conv1, self.conv2)

    def forward(self, x):
        return self.net(x)
