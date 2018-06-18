import mxnet as mx
from mxnet import nd
from mxnet import gluon


def conv(num_outputs, ks, stride, padding, bias=False):
    net = gluon.nn.Sequential()
    net.add(
        gluon.nn.Conv2D(num_outputs, ks, stride, padding),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1)
    )
    return net


def deconv(num_outputs, up=True):
    net = gluon.nn.Sequential()
    net.add(
        gluon.nn.Conv2DTranspose(num_outputs, kernel_size=4, strides=2, padding=1, weight_initializer=mx.init.Bilinear()),
        gluon.nn.LeakyReLU(0.1)
    )

    if not up:
        net[0].weight.lr_mult = 0.0

    return net


def predict_flow():
    net = gluon.nn.Sequential()
    net.add(
        # why, here the output channel is 2 !?: horizontal and vertical ?
        gluon.nn.Conv2D(channels=2, kernel_size=3, strides=1, padding=1, use_bias=False) # if using BatchNormal, can not to use bias
    )

    return net


# 由于需要结合来自底层部分的结果，所以这里就直接写在一起吧，不然不好传参数啊
# use BatchNormal
class FlowNetS(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(FlowNetS, self).__init__(**kwargs)
        # Contract Part (Encoder)
        self.conv1 = conv(num_outputs=64, ks=7, stride=2, padding=3)
        self.conv2 = conv(num_outputs=128, ks=5, stride=2, padding=2)
        self.conv3 = conv(num_outputs=256, ks=5, stride=2, padding=2)
        self.conv3_1 = conv(num_outputs=256, ks=3, stride=1, padding=1)
        self.conv4 = conv(num_outputs=512, ks=3, stride=2, padding=1)
        self.conv4_1 = conv(num_outputs=512, ks=3, stride=1, padding=1)
        self.conv5 = conv(num_outputs=512, ks=3, stride=2, padding=1)
        self.conv5_1 = conv(num_outputs=512, ks=3, stride=1, padding=1)
        self.conv6 = conv(num_outputs=1024, ks=3, stride=2, padding=1)


        # Expanding Part (Decoder)
        # For convenience, here we use Upsample('bilinear') + Convolution to replace the original 'Unpooling'
        # do not need to update the weights
        # use Bilinear() to initialize all Conv2DTranspose layers
        self.deconv5 = deconv(512)
        self.deconv4 = deconv(256)
        self.deconv3 = deconv(128)
        self.deconv2 = deconv(64)

        self.predictflow5 = predict_flow()
        self.predictflow4 = predict_flow()
        self.predictflow3 = predict_flow()
        self.predictflow2 = predict_flow()

        self.upsampled_flow_5_to_4 = gluon.nn.Conv2DTranspose(2, 4, 2, 1, use_bias=False, weight_initializer=mx.init.Bilinear())
        self.upsampled_flow_4_to_3 = gluon.nn.Conv2DTranspose(2, 4, 2, 1, use_bias=False, weight_initializer=mx.init.Bilinear())
        self.upsampled_flow_3_to_2 = gluon.nn.Conv2DTranspose(2, 4, 2, 1, use_bias=False, weight_initializer=mx.init.Bilinear())

        self.finalpredict = gluon.nn.Conv2DTranspose(2, kernel_size=8, strides=4, padding=2, weight_initializer=mx.init.Bilinear())
        self.finalpredict.weight.lr_mult = 0.0

    def forward(self, x):
        # Contracting Part
        conv_2 = self.conv2(self.conv1(x))
        conv_3 = self.conv3_1(self.conv3(conv_2))
        conv_4 = self.conv4_1(self.conv4(conv_3))
        conv_5 = self.conv5_1(self.conv5(conv_4))
        conv_6 = self.conv6(conv_5)

        # Expanding Part
        upconv5 = self.deconv5(conv_6)
        #pre_flow5 = mx.symbol.concat(conv_6, upconv5, dim=1)
        pre_flow5 = nd.concat(conv_5, upconv5, dim=1)
        flow5 = self.predictflow5(pre_flow5)

        upconv4 = self.deconv4(pre_flow5)
        #pre_flow4 = mx.symbol.concat(conv_4, upconv4, flow5, dim=1)
        upsampledflow5 = self.upsampled_flow_5_to_4(flow5)
        pre_flow4 = nd.concat(conv_4, upconv4, upsampledflow5, dim=1)
        flow4 = self.predictflow4(pre_flow4)

        upconv3 = self.deconv3(pre_flow4)
        #pre_flow3 = mx.symbol.concat(conv_3, upconv3, flow4, dim=1)
        upsampledflow4 = self.upsampled_flow_4_to_3(flow4)
        pre_flow3 = nd.concat(conv_3, upconv3, upsampledflow4, dim=1)
        flow3 = self.predictflow3(pre_flow3)

        upconv2 = self.deconv2(pre_flow3)
        upsampledflow3 = self.upsampled_flow_3_to_2(flow3)
        pre_flow2 = nd.concat(conv_2, upconv2, upsampledflow3, dim=1)
        flow2 = self.predictflow2(pre_flow2)

        #flow_output = self.finalpredict(flow2)

        return flow2, flow3, flow4, flow5


class EPError(gluon.loss.Loss):
    def __init__(self, **kwargs):
        super(EPError, self).__init__(**kwargs)

    def forward(self, pred, target, *args):
        pred = nd.flatten(pred)
        target = nd.flatten(target)

        loss = nd.norm(pred-target, 2, axis=-1, keepdims=True)

        return nd.mean(loss, axis=-1, keepdims=True)





