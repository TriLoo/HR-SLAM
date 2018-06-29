import mxnet as mx
from mxnet import gluon
from mxnet import nd


def stnlayer(data):
    #net = gluon.nn.HybridSequential()
    return mx.sym.SpatialTransformer(data, transform_type='affine')


def get_resblock_downsample(output_num):
    mid_output_num = output_num >> 1
    net = gluon.nn.HybridSequential()
    net.add(
        gluon.nn.Conv2D(output_num, 3, 1, 1),
        gluon.nn.Conv2D(mid_output_num, 1, 1),
        gluon.nn.Conv2D(output_num, 3, 1, 1)
    )

    return net


# Orange Block: ResLayer with Downsample
class resblock_downsample(gluon.nn.HybridBlock):
    def __init__(self, output_nums, **kwargs):
        super(resblock_downsample, self).__init__(**kwargs)

        mid_output_nums = output_nums >> 1

        self.conv1 = gluon.nn.Conv2D(output_nums, kernel_size=3, strides=1, padding=1)
        self.bn1 = gluon.nn.BatchNorm()
        self.actv1 = gluon.nn.Activation('relu')
        self.conv2 = gluon.nn.Conv2D(mid_output_nums, kernel_size=1, strides=1)
        self.bn2 = gluon.nn.BatchNorm()
        self.actv2 = gluon.nn.Activation('relu')
        self.conv3 = gluon.nn.Conv2D(output_nums, kernel_size=3, strides=1, padding=1)
        self.bn3 = gluon.nn.BatchNorm()
        self.actv3 = gluon.nn.Activation('relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        conv = self.actv1(self.bn1(self.conv1(x)))
        conv = self.actv2(self.bn2(self.conv1(x)))
        conv = self.actv3(self.bn3(self.conv1(x)))

        return F.relu(conv + x)


# Pink Block: ResLayer with Upsample
class resblock_upsample(gluon.nn.SymbolBlock):
    def __init__(self, num_outputs, **kwargs):
        super(resblock_upsample, self).__init__(**kwargs)

        mid_output_num = num_outputs >> 1

        self.conv1 = gluon.nn.Conv2D(num_outputs, kernel_size=3, strides=1, padding=1)
        self.bn1 = gluon.nn.BatchNorm()
        self.upconv1 = gluon.nn.Conv2DTranspose(mid_output_num, kernel_size=4, strides=2, padding=1, use_bias=False, weight_initializer=mx.init.Bilinear())
        self.upconv1_side = gluon.nn.Conv2DTranspose(mid_output_num, kernel_size=4, strides=2, padding=1, use_bias=False, weight_initializer=mx.init.Bilinear())

    def hybrid_forward(self, F, x, *args, **kwargs):
        upconv = self.upconv1(F.relu(self.bn1(self.conv1(x))))
        upconv_side = self.upconv1_side(x)

        return F.relu(upconv + upconv_side)


class ASENet(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ASENet, self).__init__(**kwargs)

        self.conv1 = gluon.nn.Conv2D(32, kernel_size=3, strides=1, padding=1)
        self.conv1_d = gluon.nn.Conv2D(32, kernel_size=3, strides=1, padding=1)

    def hybrid_forward(self, F, rgb, depth, *args, **kwargs):
        pass


class target_loss(gluon.loss.Loss):
    def __init__(self, **kwargs):
        super(target_loss, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


def train_target():
    pass
