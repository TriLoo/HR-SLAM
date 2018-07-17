# -*- coding: utf-8 -*-

__author__ = 'smh'

import mxnet as mx
from mxnet.test_utils import get_mnist_iterator
import numpy as np
import logging


class Softmax(mx.operator.CustomOp):
    # 第三输出这个
    def forward(self, is_train, req, in_data, out_data, aux):
        print('type of in_data: ', type(in_data))    # list
        print('type of out_data: ', type(in_data))   # list
        print('length of data: ', len(in_data))      # 2 ! not batch_size
        print('length of out_data: ', len(out_data))  # 1
        x = in_data[0].asnumpy()
        print('x = ', x)
        #print('in_data[1] = ', in_data[1])          # labels: shape = (batch_size, )
        print('shape of in_data[0]: ', x.shape)      # (batch_size, 10), 这个10就是fully connect的输出的大小啊
        y = np.exp(x - x.max(axis=1).reshape(x.shape[0], 1))    # involves broadcast
        y /= y.sum(axis=1).reshape(x.shape[0], 1)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    # 第四输出这个
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy()
        y = out_data[0].asnumpy()
        print('shape of l in backward: ', l.shape)    # (batch_size, ), 也就是label
        y[np.arange(l.shape[0]), 1] -= 1.0
        self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    # 先输出这个
    def infer_shape(self, in_shape):
        print('type of in_shape: ', type(in_shape))    # list
        print('length of in_shape: ', len(in_shape))   # batch_size
        print(in_shape)                                # [[batch_size, 10], [batch_size]]
        data_shape = in_shape[0]
        print('data_shape = ', data_shape)
        label_shape = (in_shape[0][0], )
        print('label_shape = ', label_shape)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    # 第二输出这个
    def infer_type(self, in_type):
        print(in_type)     # return [numpy.float32, numpy.float32]
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Softmax()


data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.sym.Activation(data = fc1, name='relu1', act_type='relu')
fc2 = mx.sym.FullyConnected(data = act1, name='fc2', num_hidden=64)
act2 = mx.sym.Activation(data = fc2, name='relu2', act_type='relu')
fc3 = mx.sym.FullyConnected(data = act2, name='fc3', num_hidden=10)

# 使用新的操作符！
mlp = mx.sym.Custom(data = fc3, name='softmax', op_type='softmax')

# data
train, val = get_mnist_iterator(batch_size=4, input_shape=(784,))

logging.basicConfig(level=logging.DEBUG)

context = mx.cpu()

mod = mx.mod.Module(mlp, context=context)

# Speedometer: Logs training speed and evaluation metrics periodically
# frequent 设置多少batch进行一次log training speed 和 evaluation metrics
# Speedometer(batch_size, frequent=50, auto_reset=True)
mod.fit(train_data=train, eval_data=val, optimizer='sgd',
        optimizer_params={'learning_rate':0.1, 'momentum':0.9, 'wd':0.00001}, num_epoch=1, batch_end_callback=mx.callback.Speedometer(4, 2))


# data's shape: (2, 784)
# softmax_label'shape (2, )
