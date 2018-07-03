import mxnet as mx
from mxnet import gluon

# Cautions:  backbone network: ResNet-50 (RedNet)

def stnlayer(data):
    #net = gluon.nn.HybridSequential()
    return mx.sym.SpatialTransformer(data, transform_type='affine')


def get_convlayer(output_num, ks, s, p=None, act=False):
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


# Green BlocK: residual layer with no downsample
class resblock_same(gluon.nn.HybridBlock):
    def __init__(self, output_nums, **kwargs):
        super(resblock_same, self).__init__(**kwargs)

        mid_output_nums = output_nums >> 2     # the first two layer in a bottleneck of resdual layer is 1/4 input channels
        self.conv1 = get_convlayer(mid_output_nums, ks=1, s=1, p=None, act=True)
        self.conv2 = get_convlayer(mid_output_nums, ks=3, s=1, p=1, act=True)
        self.conv3 = get_convlayer(output_nums, ks=1, s=1, act=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        conv1 = self.conv3(self.conv2(self.conv1(x)))

        return F.relu(x + conv1)


# Orange Block: ResLayer with Downsample
# the pattern of resnet-50
class resblock_downsample(gluon.nn.HybridBlock):
    def __init__(self, output_nums, **kwargs):
        super(resblock_downsample, self).__init__(**kwargs)

        mid_output_nums = output_nums >> 1    # output channel numbers of the first conv layer
        lst_output_nums = output_nums << 1    # output channel numbers of the last conv layer
        self.conv1 = get_convlayer(mid_output_nums, ks=1, s=1, act=True)
        self.conv2 = get_convlayer(output_nums, ks=3, s=2, p=1, act=True)         # the stride = 2, dimension halfed
        self.conv3 = get_convlayer(lst_output_nums, ks=1, s=1, act=False)
        self.conv1_side = get_convlayer(lst_output_nums, ks=1, s=2, act=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        conv1 = self.conv3(self.conv2(self.conv1(x)))
        conv1_side = self.conv1_side(x)
        return F.relu(conv1 + conv1_side)


# Pink Block: ResLayer with Upsample
class resblock_upsample(gluon.nn.SymbolBlock):
    def __init__(self, output_nums, **kwargs):
        super(resblock_upsample, self).__init__(**kwargs)

        lst_output_nums = output_nums >> 1     # the output channels is halfed

        self.conv1 = get_convlayer(output_nums, ks=3, s=1, act=True)
        self.deconv1 = get_reslayer_upsample(lst_output_nums)               # upsample the dimension & half the channels
        self.deconv1_side = get_reslayer_upsample(lst_output_nums)

    def hybrid_forward(self, F, x, *args, **kwargs):
        upconv1 = self.deconv1(self.conv1(x))
        upconv1_side = self.deconv1_side(x)

        return F.relu(upconv1 + upconv1_side)


def get_resblock_encoder(output_nums, unit_nums):
    blk = gluon.nn.HybridSequential()
    for i in range(unit_nums - 1):
        blk.add(resblock_same(output_nums))

    blk.add(resblock_downsample(output_nums))

    return blk


def get_resblock_decoder(output_nums, unit_nums):
    blk = gluon.nn.HybridSequential()
    for i in range(unit_nums-1):
        blk.add(
            resblock_same(output_nums)
        )
    blk.add(resblock_upsample(output_nums))

    return blk


class ASENet(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ASENet, self).__init__(**kwargs)

        # Encoder Part
        # covolution layer 1 with stride = 2
        self.conv1 = get_convlayer(64, ks=3, s=2, p=1, act=True)
        self.conv1_d = get_convlayer(64, ks=3, s=2, p=1, act=True)
        # max pooling layer 1 with stride = 2
        self.maxpool1 = gluon.nn.MaxPool2D(pool_size=3, strides=2)
        self.maxpool1_d = gluon.nn.MaxPool2D(pool_size=3, strides=2)
        # block layer 1 of encoder part
        self.encoder_layer1 = get_resblock_encoder(256, 3)
        self.encoder_layer1_d = get_resblock_encoder(256, 3)
        # block layer 2 of encoder part
        self.encoder_layer2 = get_resblock_encoder(512, 4)
        self.encoder_layer2_d = get_resblock_encoder(512, 4)
        # block layer 3 of encoder part
        self.encoder_layer3 = get_resblock_encoder(1024, 6)
        self.encoder_layer3_d = get_resblock_encoder(1024, 6)
        # block layer 4 of encoder part
        self.encoder_layer4 = get_resblock_encoder(2048, 3)
        self.encoder_layer4_d = get_resblock_encoder(2048, 3)

        # Decoder Part
        # No need for depth path
        self.trans1 = get_resblock_decoder(256, 6)
        self.trans2 = get_resblock_decoder(128, 4)
        self.trans3 = get_resblock_decoder(64, 3)
        self.trans4 = get_resblock_decoder(64, 3)
        self.trans5 = get_resblock_decoder(64, 3)

    def hybrid_forward(self, F, rgb, depth, *args, **kwargs):
        pass


class target_loss(gluon.loss.Loss):
    def __init__(self, **kwargs):
        super(target_loss, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


def train_target():
    pass
