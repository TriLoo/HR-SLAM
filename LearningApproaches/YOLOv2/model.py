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


class DarkNet19(gluon.nn.HybridBlock):
    def __init__(self, num_class, **kwargs):
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
                gluon.nn.MaxPool2D(strides=2),          # pool5
                conv5_6block(1024)
            )

        self.conv7 = gluon.nn.Conv2D(num_class, kernel_size=1)
        self.avgpool = gluon.nn.GlobalAvgPool2D()

        self.passthrough = self.net[-1]

        # used for detection
        self.conv8_1 = gluon.nn.Conv2D(1024, kernel_size=3, padding=1)
        self.conv8_2 = gluon.nn.Conv2D(1024, kernel_size=3, padding=1)
        self.conv8_3 = gluon.nn.Conv2D(1024, kernel_size=3, padding=1)




    def hybrid_forward(self, F, x, detection,  *args, **kwargs):
        mid_x = self.net(x)
        if detection:
            mid_x = self.conv8_1(mid_x)
            mid_x = self.conv8_2(mid_x)
            mid_x = self.conv8_3(mid_x)
            y = self.conv7(mid_x)

            #y = mx.symbol.softmax(mid_x)

        else:
            mid_x = self.conv7(mid_x)
            y = self.avgpool(mid_x)

        # OR
        #return mx.symbol.softmax(y)
        return nd.Softmax(y)


