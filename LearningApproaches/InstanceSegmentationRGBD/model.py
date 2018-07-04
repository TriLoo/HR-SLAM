import mxnet as mx
from mxnet import gluon

# 参考文献：RedNet: Residual Encoder-Decoder Network for indoor RGB-D Semantic Segmentation, 2018.06.04 arxiv
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
# 即普通的具有bottleneck结构的res block
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


# 文章说了，在encoder部分的layer1~4里面，每层layer只有一层是具有降维功能的res layer with downsample， 其他层是具有bottleneck的res block(见ResNet原文)
# 注意，在每一个layer中，模型是先进行降维(一层res layer with downsample)，然后后面在跟几个不降维的res block (unit)
def get_resblock_encoder(output_nums, unit_nums, down=True):
    blk = gluon.nn.HybridSequential()

    if down:       # encoder中的除了第一层layer之外的layer的结构，先进行降维，然后后面跟着几个具有bottleneck的res block
        blk.add(resblock_downsample(output_nums))       # 利用res layer with downsample首先进行降维
        for i in range(unit_nums - 1):
            blk.add(resblock_same(output_nums))
    else:                                               # 这里主要用于实现encoder里面的第一层layer，即不降维
        for i in range(unit_nums):                      # 不降维的话，就是普通的绿色block, 即全是具有bottleneck的res block
            blk.add(resblock_same(output_nums))

    return blk


# 文章说了，在decoder部分的trans1-5里面，每层trans只有一层是具有升维功能的res layer with upsample， 其它层是具有普通两层3*3卷积同等输出channel的res block.
# 在decoder的最后一层是一层Conv2DTranspose实现，而不是包含res layer with upsample的trans层实现的
# 注意，与encoder部分不同的是，在每一个trans结构里面，是先存在几个不降维的res block (unit)，在最后才会跟一层 res layer with upsample
def get_resblock_decoder(output_nums, unit_nums):
    blk = gluon.nn.HybridSequential()
    for i in range(unit_nums-1):
        blk.add(
            resblock_same(output_nums)     # 这里与原文不同，原文这里并没有采用bottleneck结构，而是普通的两层3*3同等输出channel数量的卷积层，见ResNet论文,为了方便，这里采用bottleneck 结构
        )
    blk.add(resblock_upsample(output_nums))

    return blk


# 这里采用的是res-50结构
# 在encoder部分共降维32倍，在decoder部分共升维32倍。
# 在res-50作为encoder时，采用了channel扩充的思想，也就是在encoder部分的channel数量比较大；为了降低存储需求，所以在decoder部分采用的channel数量就变得很小。
# 由于存在encoder到decoder部分的pass连接，但二者的channel数量不同，所以模型中又加入了Agent(1*1的卷积核)结构用于对encoder部分进行project到较少channel的数量
# 在encoder里面的layer1是没有降维的！这一块是配合前一层模型中唯一的max pooling with stride=2来实现的降维，文章中这一层layer是绿色的，说明不降维，橘黄色的encoder layer是自带降维
# 所有的fusion操作是 Element-wise Addition
class ASENet(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ASENet, self).__init__(**kwargs)

        # Encoder Part
        # covolution layer 1 with stride = 2
        self.conv1 = get_convlayer(64, ks=7, s=2, p=3, act=True)
        self.conv1_d = get_convlayer(64, ks=7, s=2, p=3, act=True)
        # max pooling layer 1 with stride = 2
        self.maxpool1 = gluon.nn.MaxPool2D(pool_size=3, strides=2)
        self.maxpool1_d = gluon.nn.MaxPool2D(pool_size=3, strides=2)
        # block layer 1 of encoder part
        self.encoder_layer1 = get_resblock_encoder(256, 3, down=False)          # 这里配合上面的maxpool1(_d)实现降维
        self.encoder_layer1_d = get_resblock_encoder(256, 3, down=False)
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
        # TODO: construct the whole network basing on above calss variables
        pass


class target_loss(gluon.loss.Loss):
    def __init__(self, **kwargs):
        super(target_loss, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


def train_target():
    pass
