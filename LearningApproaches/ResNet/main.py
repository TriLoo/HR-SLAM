import mxnet as mx
import model as res

net = res.ResNet()
net.initialize(init=mx.init.Xavier)



