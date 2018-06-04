import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import nd


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


class DarkNet19(gluon.nn.HybridBlock):
    def __init__(self, num_class, **kwargs):   # for VOC: num_class = 125: 5 * (120 + 5)
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
        self.outputlayer = gluon.nn.Conv2D(num_class, kernel_size=1)
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
        else:
            mid_x = self.pool5(mid_x)
            mid_x = self.conv6(mid_x)
            mid_x = self.outputlayer(mid_x)
            y = self.avgpool(mid_x)
'''
        # OR
        #return nd.Softmax(y)


