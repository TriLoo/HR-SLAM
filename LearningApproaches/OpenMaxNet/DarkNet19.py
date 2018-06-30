import mxnet as mx
from mxnet.gluon import model_zoo
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import readCUBData
from mxnet import metric

# use resnet to be the model
# pretrained_resnet18 = model_zoo.vision.resnet18_v2(classes=100)
# the fully connected input size is
pretrained_resnet18 = model_zoo.vision.resnet18_v2(pretrained = True, ctx=mx.cpu())

finetune_resnet18 = model_zoo.vision.resnet18_v2(classes=100, ctx=mx.gpu())
finetune_resnet18.features = pretrained_resnet18.features
finetune_resnet18.output.initialize(mx.init.Xavier())


#cls_acc = metric.Accuracy()
def accuracy(output, label):
    return nd.mean(output.argmax(axis=-1)==label).asscalar()


'''
def evaluate_accuracy(data_iterator, net, batch_size=2, ctx=[mx.gpu()]):
    if isinstance(ctx, mx.Context):
        ctx=[ctx]

    acc = nd.array)[0]
    n = 0.

    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label =
'''


# default loss = SoftmaxCrossEntropy()
def train(net, train_data, test_data, lossfunc, learning_rate, batch_size, ctx=mx.gpu(), epoches=30):
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':learning_rate, 'wd':0.001, 'momentum':0.01})
    train_loss = []
#cls_acc.reset()
    for epoch in range(epoches):
#train_loss = 0.0
        for i, batch in enumerate(train_data):
            data, label = batch
            data = data.copyto(mx.gpu())
            label = label.copyto(mx.gpu())
            with autograd.record():
                output = net(data)                    # list
                currloss = lossfunc(output, label)    # hybrid_forward(F, pred, label, ...)
#print('shape of output = ', len(output))           # output: 2
            currloss.backward()
            trainer.step(batch_size)
#train_loss += sum([l.sum().asscalar() for l in currloss])
            #print('type of currloss: {}, shape of currloss: {}'.format(type(currloss), currloss.shape), currloss)
            if (i) % 100 == 0:
                train_loss.append(currloss)    
                print('current loss = ', nd.sum(currloss).asscalar())

        print('Epoch: %d, training loss: ' % (epoch), nd.sum(train_loss[-1]).asscalar())






if __name__ == '__main__':
    '''
    Flow: read data, create model, initialize, 
            create loss fun, training
    x = nd.random_uniform(0, 1, shape=(2, 3, 312, 320))
    y = finetune_resnet18(x) # Input: (1, 100)
    print(y.shape)   # (2, 100)
    '''
    data_dir = '/home/slz/.mxnet/datasets/CUB100'
    net = pretrained_resnet18
    lr = 0.15
    lossfunc = gluon.loss.SoftmaxCrossEntropyLoss()
    batch_size = 8
    train_set = gluon.data.vision.ImageFolderDataset(data_dir, transform=lambda X, y : readCUBData.transform(X, y, readCUBData.augs))
    train_data = gluon.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train(net, train_data, None, lossfunc, lr, batch_size)

    try:
        net.collect_params().save('resnet.params')
    except:
        print('save failed')



'''
y = finetune_resnet18(x)
print(y.shape)    # output: (1, 100)
'''


'''
class YOLO2Output(gluon.nn.HybridSequential):
    def __init__(self, num_classes, anchor_scales, **kwargs):
        super(YOLO2Output, self).__init__(**kwargs)
        assert num_classes>0, "number of classes should > 0, given{}".format(num_classes)

        self._num_class = num_classes
        assert isinstance(anchor_scales, (list, tuple)), 'list or tuple of anchor scales required'
        assert len(anchor_scales) > 0, 'at least one anchor scale required'

        for anchor in anchor_scales:
            assert len(anchor) == 2, 'expected each anchor scale to be (width, height), provided {}'.format(anchor)

        self._anchor_scales = anchor_scales
        out_channels = len(anchor_scales) * (num_classes + 1 + 4)
        with self.name_scope():
            self.output = gluon.nn.Conv2D(out_channels, 1, 1)

    def hybrid_forward(self, F, x):
        return self.output(x)



def conv_at_layer(in_data, kernel, stride, num_filter, activation='leaky', padding=1, use_bn = True):
    out_data = mx.symbol.Convolution(in_data, kernel=kernel, stride=stride, pad=padding, num_filter=num_filter)
    # use BN before the activation
    if use_bn:
        out_data = mx.symbol.BatchNorm(out_data)
    out_data = mx.symbol.LeakyReLU(out_data, act_type=activation)

    return out_data


# Construct the blocks using Function
def conv3_4block(output_channel):
    mid_channel = output_channel // 2
    blk = gluon.nn.HybridSequential()
    blk.add(
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(mid_channel, 1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.MaxPool2D(strides=2)
    )

    return blk


def conv5_6block(output_channel):
    mid_channel = output_channel // 2
    blk = gluon.nn.HybridSequential()
    blk.add(
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(mid_channel, 1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(mid_channel, 1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
    )

    return blk


# reshape the 26 * 26 * 512 to 13 * 13 * 2048
def stack_neightbor(in_data, factor=2):
    # output: (batch, channel, h/2, w)
    out = mx.symbol.reshape(in_data, shape=(0, 0, -4, -1, factor, -2)) # -4后面的两个参数表明h维被分割成h/2和2(factor)了
    out = mx.symbol.transpose(out, axes=(0, 1, 3, 2, 4))
    out = mx.symbol.reshape(out, shape=(0, -3, -1, -2))
    out = mx.symbol.reshape(out, shape=(0, 0, 0, -4, -1, factor))
    out = mx.symbol.transpose(out, axes=(0, 1, 4, 2, 3))
    out = mx.symbol.reshape(out, shape=(0, -3, -1, -2))

    return out


# Used for classification
class DarkNet19(gluon.nn.HybridBlock):
    def __init__(self, num_class, anchor_scales, **kwargs):   # for VOC: num_class = 125: 5 * (120 + 5)
        super(DarkNet19, self).__init__(**kwargs)
        self.net = gluon.nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(32, 3, padding=1),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.1),                # conv1, Set the alpha to 0.1
                gluon.nn.MaxPool2D(strides=2),          # pool1
                gluon.nn.Conv2D(64, 3, padding=1),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.1),                # conv2 end
                gluon.nn.MaxPool2D(strides=2),          # pool2
                conv3_4block(128),                      # conv3 + pool3
                conv3_4block(256),                      # conv4 + pool4
                conv5_6block(512),                      # conv5
            )

        self.pool5 = gluon.nn.MaxPool2D(strides=2)          # pool5
        self.conv6 = conv5_6block(1024)

        self.output_nums = len(anchor_scales) * (num_class + 1 + 4)
        #self.outputlayer = gluon.nn.Conv2D(self.output_nums, kernel_size=1)
        self.outputlayer = YOLO2Output(num_class, anchor_scales)


        self.avgpool = gluon.nn.GlobalAvgPool2D()

        self.passthrough = self.net[-1]

        # used for detection
        self.conv8 = gluon.nn.HybridSequential()
        self.conv8.add(
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1),
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1),
        )

        self.conv9 = gluon.nn.HybridSequential()
        self.conv9.add(
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1)
        )


    #def hybrid_forward(self, F, x, detection = True,  *args, **kwargs):
    def hybrid_forward(self, F, x,  *args, **kwargs):
        mid_x = self.net(x)
        #if detection:
        mid_y = self.pool5(mid_x)
        mid_y = self.conv6(mid_y)
        mid_y = self.conv8(mid_y)
        mid_y = mx.symbol.concat(mid_y, stack_neightbor(mid_x), dim=1)
        mid_y = self.conv9(mid_y)
        y = self.outputlayer(mid_y)

        return mx.symbol.softmax(y)

 Used for detection
class DarkNet19(gluon.nn.HybridBlock):
    def __init__(self, num_class, anchor_scales, **kwargs):   # for VOC: num_class = 125: 5 * (120 + 5)
        super(DarkNet19, self).__init__(**kwargs)
        self.net = gluon.nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(32, 3, padding=1),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.1),                # conv1, Set the alpha to 0.1
                gluon.nn.MaxPool2D(strides=2),          # pool1
                gluon.nn.Conv2D(64, 3, padding=1),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.1),                # conv2 end
                gluon.nn.MaxPool2D(strides=2),          # pool2
                conv3_4block(128),                      # conv3 + pool3
                conv3_4block(256),                      # conv4 + pool4
                conv5_6block(512),                      # conv5
            )

        self.pool5 = gluon.nn.MaxPool2D(strides=2)          # pool5
        self.conv6 = conv5_6block(1024)

        self.output_nums = len(anchor_scales) * (num_class + 1 + 4)
        #self.outputlayer = gluon.nn.Conv2D(self.output_nums, kernel_size=1)
        self.outputlayer = YOLO2Output(num_class, anchor_scales)


        self.avgpool = gluon.nn.GlobalAvgPool2D()

        self.passthrough = self.net[-1]

        # used for detection
        self.conv8 = gluon.nn.HybridSequential()
        self.conv8.add(
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1),
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1),
        )

        self.conv9 = gluon.nn.HybridSequential()
        self.conv9.add(
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1)
        )


    #def hybrid_forward(self, F, x, detection = True,  *args, **kwargs):
    def hybrid_forward(self, F, x,  *args, **kwargs):
        mid_x = self.net(x)
        #if detection:
        mid_y = self.pool5(mid_x)
        mid_y = self.conv6(mid_y)
        mid_y = self.conv8(mid_y)
        mid_y = mx.symbol.concat(mid_y, stack_neightbor(mid_x), dim=1)
        mid_y = self.conv9(mid_y)
        y = self.outputlayer(mid_y)

        return mx.symbol.softmax(y)
'''
