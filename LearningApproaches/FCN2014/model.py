import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as modelv
from mxnet.gluon import utils as gutils
from time import time

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]

    filt = (1 - abs(og[0] - center)/factor) * (1 - abs(og[1] - center)/factor)

    weight = np.zeros(shape=(in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt

    return nd.array(weight)


class FCNet(gluon.nn.HybridBlock):
    def __init__(self, class_num, **kwargs):
        super(FCNet, self).__init__(**kwargs)
        pretrained_net = modelv.resnet18_v2(ctx=mx.gpu(), pretrained=True)

        self.net = gluon.nn.HybridSequential()
        for layer in pretrained_net.features[:-2]:
            with self.net.name_scope():     # add name_scope()
                self.net.add(layer)

        with self.net.name_scope():
            self.net.add(gluon.nn.Conv2D(channels=class_num, kernel_size=1),
                         gluon.nn.Conv2DTranspose(channels=class_num, kernel_size=64, padding=16, strides=32))

        self.net[-2].initialize(init=mx.init.Xavier(), ctx=mx.gpu())
        self.net[-1].initialize(init=mx.init.Bilinear(), ctx=mx.gpu())
        #self.net[-1].weight.lr_mult=0

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.net(x)      # OR:
        #return nd.softmax(self.net(x), axis=1)


def _get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):   # in fact, the batch is a list
        features = batch.data[0]
        labels = batch.label[0]
    else:
        features, labels = batch

    # use split_and_load: split the data(ndarray) to len(ctx_list) slices along batch_axis, and loads each slice to one context in ctx_list
    # here, the len(ctx_list) = 0, return data.as_in_context(ctx), this can be found in the source of this function.
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])   # return the batch size


def test_accuracy(test_data, net, ctx=mx.gpu()):
    acc = nd.array([0])
    n = 0
    for batch in test_data:    # batch.data.shape = (batch_size, 3, 320, 480), type(batch) = list
        features, labels, batch_size = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size

        acc.wait_to_read()
        return acc.asscalar() / n


def train(net, train_data, test_data, loss, trainer, ctx = mx.gpu(), num_epochs = 1000, verbose=True):
    net.collect_params().reset_ctx(ctx)
    for epoch in range(1, num_epochs+1):
        train_l_sum, train_acc_sum, n, m = .0, .0, .0, .0
        start = time()
        for i, batch in enumerate(train_data):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]

            for l in ls:
                l.backward()

            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
            train_l_sum += sum([l.sum().asscalar() for l in ls])

            trainer.step(batch_size)

            n += batch_size
            m += sum(y.size for y in ys)

        test_acc = test_accuracy(test_data, net, ctx)
        if verbose:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'%(epoch, train_l_sum/n, train_acc_sum/m, test_acc, time() - start))


def predict(net, im, ctx, func):
    data = func(im)
    data = data.transpose((2, 0, 1).expand_dims(axis=0))
    yhat = net(data.as_in_context(ctx))
    pred = nd.argmax(yhat, axis = 1)
    return pred.reshape((pred.shape[1], pred.shape[2]))
