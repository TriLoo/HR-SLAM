import mxnet as mx
from mxnet.gluon import model_zoo
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import readCUBData
from mxnet import metric

# use resnet to be the model
# pretrained_resnet18 = model_zoo.vision.resnet18_v2(classes=100)
pretrained_resnet18 = model_zoo.vision.resnet18_v2(pretrained = True, ctx=mx.cpu())

finetune_resnet18 = model_zoo.vision.resnet18_v2(classes=100, ctx=mx.gpu())
finetune_resnet18.features = pretrained_resnet18.features
finetune_resnet18.output.initialize(mx.init.Xavier())

cls_acc = metric.Accuracy()


def train(net, train_data, test_data, lossfunc, learning_rate, batch_size, ctx=mx.gpu(), epoches=30):
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':learning_rate, 'wd':0.001, 'momentum':0.01})
    train_loss = []
    cls_acc.reset()
    for epoch in range(epoches):
        for i, batch in enumerate(train_data):
            data, label = batch
            data = data.copyto(mx.gpu())
            label = label.copyto(mx.gpu())
            with autograd.record():
                output = net(data)                    # list
                currloss = lossfunc(output, label)    # hybrid_forward(F, pred, label, ...)
            currloss.backward()
            trainer.step(batch_size)
            cls_acc.update(label, output.argmax(axis=1))    # (labels, preds)
            if (i) % 100 == 0:
                train_loss.append(currloss)
                print('current loss = ', nd.sum(currloss).asscalar())

        print('Epoch: %d, training acc: %s %.2f' % (epoch), *cls_acc.get())


if __name__ == '__main__':
    '''
    Flow: read data, create model, initialize, 
            create loss fun, training
    '''
    data_dir = '/home/slz/.mxnet/datasets/CUB100'
    net = finetune_resnet18
    lr = 0.1
    lossfunc = gluon.loss.SoftmaxCrossEntropyLoss()
    batch_size = 8
    train_set = gluon.data.vision.ImageFolderDataset(data_dir, transform=lambda X, y : readCUBData.transform(X, y, readCUBData.augs))
    train_data = gluon.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train(net, train_data, None, lossfunc, lr, batch_size)

    try:
        net.collect_params().save('resnet.params')
    except:
        print('save failed')

