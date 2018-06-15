from mxnet import gluon
import numpy as np
from mxnet import nd


data_in = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(data_in)

kernel = nd.array([[0, 1], [2, 3]])

net = gluon.nn.Sequential()
net.add(gluon.nn.Conv2D(channels=1, kernel_size=(3, 3), strides=(1, 1)),
        gluon.nn.MaxPool2D())

net.initialize()

x = nd.random_uniform(shape=(1, 1, 10, 10))
y = net(x)
print(net[0].params)

print(net[0].weight)

print(net[0].weight.data())

print('x = ', x)
print('y = ', y)

Decov = gluon.nn.Sequential()
Decov.add(gluon.nn.Conv2DTranspose(channels=1, kernel_size=(4, 4), strides=(2, 2)))

Decov.initialize()
y_ = Decov(y)
print('y_ = ', y_)


