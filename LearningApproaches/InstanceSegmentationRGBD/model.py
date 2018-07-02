import mxnet as mx
from mxnet import gluon

# Cautions:  backbone network: ResNet-50 (RedNet)

def stnlayer(data):
    #net = gluon.nn.HybridSequential()
    return mx.sym.SpatialTransformer(data, transform_type='affine')


def get_reslayer_downsample(output_num, ks, s, p=None, act=False):
    net = gluon.nn.HybridSequential()
    net.add(
        gluon.nn.Conv2D(output_num, kernel_size=ks, strides=s, padding=p),
        gluon.nn.BatchNorm()
    )

    if act:
        net.add(gluon.nn.Activation('relu'))

    return net


def get_reslayer_upsample(output_num, ks=4, s=2, p=1, act=False):
    net = gluon.nn.HybridSequential()
    net.add(
        gluon.nn.Conv2DTranspose(output_num, kernel_size=ks, strides=s, padding=p),
        gluon.nn.BatchNorm()
    )

    if act:
        net.add(gluon.nn.Activation('relu'))

    return net


# Orange Block: ResLayer with Downsample
# the pattern of resnet-50
class resblock_downsample(gluon.nn.HybridBlock):
    def __init__(self, output_nums, **kwargs):
        super(resblock_downsample, self).__init__(**kwargs)

        mid_output_nums = output_nums >> 1    # output channel numbers of the first conv layer
        lst_output_nums = output_nums << 1    # output channel numbers of the last conv layer
        self.conv1 = get_reslayer_downsample(mid_output_nums, ks=1, s=1, act=True)
        self.conv2 = get_reslayer_downsample(output_nums, ks=3, s=2, p=1, act=True)         # the stride = 2, dimension halfed
        self.conv3 = get_reslayer_downsample(lst_output_nums, ks=1, s=1, act=False)
        self.conv1_side = get_reslayer_downsample(lst_output_nums, ks=1, s=2, act=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        conv1 = self.conv3(self.conv2(self.conv1(x)))
        conv1_side = self.conv1_side(x)
        return F.relu(conv1 + conv1_side)


# Pink Block: ResLayer with Upsample
class resblock_upsample(gluon.nn.SymbolBlock):
    def __init__(self, output_nums, **kwargs):
        super(resblock_upsample, self).__init__(**kwargs)

        lst_output_nums = output_nums >> 1     # the output channels is halfed

        self.conv1 = get_reslayer_downsample(output_nums, ks=3, s=1, act=True)
        self.deconv1 = get_reslayer_upsample(lst_output_nums)               # upsample the dimension & half the channels
        self.deconv1_side = get_reslayer_upsample(lst_output_nums)

    def hybrid_forward(self, F, x, *args, **kwargs):
        upconv1 = self.deconv1(self.conv1(x))
        upconv1_side = self.deconv1_side(x)

        return F.relu(upconv1 + upconv1_side)


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
