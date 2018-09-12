# -*- coding: utf-8 -*-

'''
Spatial-Temporal Attention Module

Refernece:
    * CBAM: Convolutional Block Attention Module
    * BAM: Bottleneck Attention Module
    SENet, BiSeNet
    ** RecalibratingFCNsWithSpatialAndChannelSEBlocks
    Residual Attention Networks For Image Classification

    - - - -
    mxnet.symbol:
    refs: https://mxnet.apache.org/api/python/module/module.html
          https://mxnet.incubator.apache.org/tutorials/basic/module.html
          https://discuss.mxnet.io/t/initialize-weights-with-predefined-values/80/3

'''

__author__ = 'smh'
__date__ = '2018.08.28'

import mxnet as mx
import numpy as np
from mxnet import nd


# channel Squeeze & spatial Excitation
def channelS_spatialE(data):
    spatial_weigth = mx.sym.max(data, axis=1, name='csse_maxpooling')
    spatial_scale = mx.sym.Activation(spatial_weigth, act_type='sigmoid', name='csse_sigmoid')
    csse = mx.sym.broadcast_mul(lhs=data, rhs=spatial_scale, name='csse_broad_mult')
    return csse


# channel Excitation & spatial Squeeze
def channelE_spatialS(data, fn):
    #fc0_weight = mx.sym.Variable('fc0_weight')
    fc1_weight = mx.sym.Variable('fc1_weight')
    fc2_weight = mx.sym.Variable('fc2_weight')
    global_p = mx.sym.Pooling(data, pool_type='max', global_pool=True, name='cess_maxpooling')
    #channel_weight1_conv = mx.sym.FullyConnected(global_p, num_hidden=fn, weight=fc0_weight, no_bias=True, flatten=True, name='cess_fc0_conv')
    channel_weight1_conv = mx.sym.FullyConnected(global_p, num_hidden=fn, no_bias=True, flatten=True, name='cess_fc0_conv')
    channel_weight1_bn = mx.sym.BatchNorm(channel_weight1_conv, name='cess_fc0_bn')
    channel_weight1_relu = mx.sym.Activation(channel_weight1_bn, act_type='relu', name='cess_fc0_relu')
    channel_weight2_conv = mx.sym.FullyConnected(channel_weight1_relu, num_hidden=(fn >> 1), weight=fc1_weight, no_bias=True, flatten=True, name='cess_fc1_conv')
    channel_weight2_bn = mx.sym.BatchNorm(channel_weight2_conv, name='cess_fc1_bn')
    channel_weight2_relu = mx.sym.Activation(channel_weight2_bn, act_type='relu', name='cess_fc1_relu')
    channel_weight3_conv = mx.sym.FullyConnected(channel_weight2_relu, num_hidden=fn, flatten=True, weight=fc2_weight, no_bias=True, name='cess_fc2_conv')
    channel_scale = mx.sym.Activation(channel_weight3_conv, act_type='sigmoid', name='cess_fc2_sigmoid')
    cess = mx.sym.broadcast_mul(lhs=data, rhs=mx.sym.reshape(data=channel_scale, shape=(-1, fn, 1, 1)), name='cess_broad_mult')
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
    cn = 1
    print('shape of data: ', data.shape)
    #output_cs_se = channelS_spatialE(data)
    a = mx.sym.Variable('a')
    csse = channelE_spatialS(a, 3)
    print('input of csse: ', csse.list_arguments())
    #print('output shape of csse: ', csse.infer_shape(a=(1, 3, 10, 10)))
    print('type of cess: ', type(csse))
    mod = mx.mod.Module(csse, context=mx.cpu(), data_names=['a'], label_names=[])
    print('type of data type: ', type(data.shape))    # tuple
    mod.bind(data_shapes=[('a', data.shape)], for_training=False)   # Note here
    mod.init_params(initializer=mx.init.Xavier())
    mod.forward(mx.io.DataBatch([data]))
    output_mod = mod.get_outputs()[0].asnumpy()
    print(type(output_mod))
    print(output_mod.shape)
    #mod.predict(eval_data=data)

    '''
    ex = csse.simple_bind(ctx=mx.cpu(), data=data, fn=3)
    #ex = csse.simple_bind(ctx=mx.cpu(), a=(1, 3, 10, 10), fn=3)
    ex.forward()
    out_val = ex.outputs[0].asnumpy()
    print('output shape of sym calculation: ', out_val.shape)
    print(out_val)
    '''
