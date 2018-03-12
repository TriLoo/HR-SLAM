import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd

class SVS(nn.Block):
    def __init__(self, **kwargs):
        super(SVS, self).__init__(**kwargs)

    def forward(self, *args):
        return x - x.mean()
        
