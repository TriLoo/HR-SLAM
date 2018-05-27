import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import numpy as np
from mxnet import nd

class AlexNet(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(AlexNet, self).__init__(**kwargs)

        self.net = gluon.nn.Sequential()
        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=3, strides=2),
                gluon.nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=3, strides=2),
                gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
                gluon.nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
                gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
                gluon.nn.MaxPool2D(pool_size=3, strides=2),
                gluon.nn.Dense(2048, activation='relu'),    # reduce the number of parameters
                gluon.nn.Dropout(0.5),
                gluon.nn.Dense(2048, activation='relu'),
                gluon.nn.Dropout(0.5),
                gluon.nn.Dense(10)
            )

    def forward(self, x):
        return nd.softmax(self.net(x))


def train(net, train_set, test_set, lossFunc,  batchSize, lr, epochs, period, eps = 1e-6, verbose=False):
    assert batchSize >= period and period % batchSize == 0
    trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate':lr, 'beta1':0.9, 'beta2':0.999})
    total_loss = []

    train_data = gluon.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)

    for epoch in range(epochs):
        for batch_i, (data, label) in enumerate(train_data):
            #print('Traning %d'%(batch_i))
            data = data.as_in_context(mx.gpu())
            label = label.as_in_context(mx.gpu())
            with autograd.record():
                output = net(data)
                loss = lossFunc(output, label)
            loss.backward()
            trainer.step(batchSize)

            if batch_i * batchSize % period == 0:
                total_loss.append(np.mean(lossFunc(net(data).as_in_context(mx.cpu()), label.as_in_context(mx.cpu())).asnumpy()))

        if verbose:
            print('Batch Size: %d, learning rate: %f, epoch: %f, loss %.4e'%(batchSize, lr, epoch, total_loss[-1]))

        if epoch % 200 == 0:
            lr = lr / 10

        if lr < 0.000001:
            lr = 0.000001

        #if len(total_loss) > 2 and total_loss[-2] - total_loss[-1] < eps:
            #break

    return total_loss


def predict(net, data, verbose = True):
    pred = []
    #for i, data in enumerate(datas):
    data = data.as_in_context(mx.gpu())
    label = net(data)
    pred.append(label.as_in_context(mx.cpu()))

    if verbose:
        print('image, predicted label = %d'%(pred[-1]))

    return pred

