import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet import autograd
import model

net = model.FlowNetS()
net.initialize(init=mx.init.Xavier())

x = nd.ones((1, 6, 384, 512))
y = net(x)

print(y[0].shape)    # output: (1, 2, 96, 128)
print(y[1].shape)    # output: (1, 2, 48, 64)
print(y[2].shape)    # output: (1, 2, 24, 32)
print(y[3].shape)    # output: (1, 2, 12, 16)

