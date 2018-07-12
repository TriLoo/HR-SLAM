# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.07.12'

import mxnet as mx
import model
from mxnet import nd
import argparse

'''
x = nd.random_uniform(0, 1, shape=(1, 3, 320, 1024))

net = model.resblock_same(256)
net.initialize(mx.init.Xavier())
net.hybridize()
rgb = nd.random_uniform(0, 1, shape=(1, 256, 512, 512))
y = net(rgb)
print(y.shape)
'''

'''
# Test the model
net = model.ASENet(10)
net.initialize(init=mx.init.Xavier())
net.hybridize()
#print(net)

rgb = nd.random_uniform(0, 1, shape=(1, 3, 512, 512))
depth = nd.random_uniform(0, 1, shape=(1, 1, 512, 512))

a, b, c, d, e = net(rgb, depth)
print('a.shape = ', a.shape)
print('b.shape = ', a.shape)
print('c.shape = ', a.shape)
print('d.shape = ', a.shape)
print('e.shape = ', a.shape)
'''
'''
net = model.resblock_downsample(256)
net.initialize(mx.init.Xavier())
net.hybridize()

x = nd.random_uniform(0, 1, (1, 64, 512, 512))

y = net(x)

print(y.shape)
'''

'''
net = model.resblock_upsample(256)
net.initialize(mx.init.Xavier())
net.hybridize()

x = nd.random_uniform(0, 1, (1, 512, 128, 128))
y = net(x)
print(y.shape)
'''

'''
net = model.get_resblock_encoder(512, 4, True)
net.initialize(mx.init.Xavier())
net.hybridize()

x = nd.random_uniform(0, 1, (1, 256, 128, 128))
y = net(x)
print(y.shape)
'''

'''
net = model.get_resblock_encoder(256, 3, False)
net.initialize(mx.init.Xavier())
net.hybridize()

x = nd.random_uniform(0, 1, (1, 64, 128, 128))
y = net(x)
print(y.shape)
'''

'''
#net = model.get_resblock_decoder(256, 6, True)
net = model.get_resblock_decoder(256, 6, False)
net.initialize(mx.init.Xavier())
net.hybridize()

x = nd.random_uniform(0, 1, (1, 512, 64, 64))
y = net(x)
print(y.shape)
'''

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoches', type=int, default=10)
    args = parse.parse_args()
    net = model.ASENet(10)
    net.initialize(mx.init.Xavier())
    net.hybridize()
    #print(net)

    rgb = nd.random_uniform(0, 1, (1, 3, 512, 512))
    depth = nd.random_uniform(0, 1, (1, 1, 512, 512))

    y1, y2, y3, y4, y5 = net(rgb, depth)
    #y1, y2 = net(rgb, depth)

    print('y1.shape: ', y1.shape)
    print('y2.shape: ', y2.shape)
    print('y3.shape: ', y3.shape)
    print('y4.shape: ', y4.shape)
    print('y5.shape: ', y5.shape)


if __name__ == '__main__':
    main()


