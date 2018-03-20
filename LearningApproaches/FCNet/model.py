import mxnet 
from mxnet import autograd
from mxnet import gluon

'This file includes the implementation of <Full Convolutional Networks for Semantic Segmentation>'

class FCNet(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(FCNet, self).__init__(**kwargs)

    def forward(self, x):
        return x

