# -*- coding: utf-8 -*-

'''
Spatial-Temporal Attention Module

Refernece:
    * CBAM: Convolutional Block Attention Module
    * BAM: Bottleneck Attention Module
    SENet, BiSeNet
    ** RecalibratingFCNsWithSpatialAndChannelSEBlocks
    Residual Attention Networks For Image Classification
'''

__author__ = 'smh'
__date__ = '2018.08.28'

import mxnet as mx
import numpy as np
from mxnet import nd


# channel Squeeze & spatial Excitation
def channelS_spatialE(data):
    spatial_weigth = mx.sym.max(data, axis=1)
    spatial_scale = mx.sym.Activation(spatial_weigth, act_type='sigmoid')
    csse = mx.sym.broadcast_mul(lhs=data, rhs=spatial_scale, name='csse')
    return csse


# channel Excitation & spatial Squeeze
def channelE_spatialS(data, fn):
    global_p = mx.sym.Pooling(data, pool_type='max', global_pool=True)
    channel_weight1_conv = mx.sym.FullyConnected(global_p, num_hidden=fn, flatten=True)
    channel_weight1_bn = mx.sym.BatchNorm(channel_weight1_conv)
    channel_weight1_relu = mx.sym.Activation(channel_weight1_bn, act_type='relu')
    channel_weight2_conv = mx.sym.FullyConnected(channel_weight1_relu, num_hidden=fn*0.5, flatten=True)
    channel_weight2_bn = mx.sym.BatchNorm(channel_weight2_conv)
    channel_weight2_relu = mx.sym.Activation(channel_weight2_bn, act_type='relu')
    channel_weight3_conv = mx.sym.FullyConnected(channel_weight2_relu, num_hidden=fn, flatten=True)
    channel_scale = mx.sym.Activation(channel_weight3_conv, act_type='sigmoid')
    cess = mx.sym.broadcast_mul(lhs=data, rhs=channel_scale, name='cess')
    return cess


if __name__ == '__main__':
    # Test channelS_spatialE
    a = np.ones(shape=(1, 10, 10)) * 2
    b = []
    b.append(a)
    a = np.ones(shape=(1, 10, 10))
    b.append(a)
    a = np.ones(shape=(1, 10, 10))
    b.append(a)
    data = nd.array(b)
    data = nd.transpose(data, axes=(1, 0, 2, 3))
    print('shape of data: ', data.shape)
    #output_cs_se = channelS_spatialE(data)
    a = mx.sym.Variable('a')
    csse = channelS_spatialE(a)
    ex = csse.bind(ctx=mx.cpu(), args={'a':data})
    ex.forward()
    out_val = ex.outputs[0].asnumpy()
    print('output shape of sym calculation: ', out_val.shape)
    print(out_val)

