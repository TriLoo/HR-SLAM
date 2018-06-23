import mxnet as mx
from mxnet import gluon
from mxnet.ndarray.contrib import BilinearResize2D


def conv(num_outputs, ks, stride, padding):
    net = gluon.nn.HybridSequential()
    net.add(
        gluon.nn.Conv2D(num_outputs, ks, stride, padding),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1)
    )

    return net


def deconv(num_outputs):
    net = gluon.nn.HybridSequential()
    net.add(
        gluon.nn.Conv2DTranspose(num_outputs, kernel_size=4, strides=2, padding=1, use_bias=False, weight_initializer=mx.initializer.Bilinear()),
        gluon.nn.LeakyReLU(0.1)
    )

    return net


def predict_flow():
    net = gluon.nn.HybridSequential()
    net.add(
        gluon.nn.Conv2D(channels=3, kernel_size=3, strides=1, padding=1, use_bias=False)
    )
    return net


class FlowNetS(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(FlowNetS, self).__init__(**kwargs)

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

        self.upsampled_flow_5_to_4 = gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False, weight_initializer=mx.init.Bilinear())
        self.upsampled_flow_4_to_3 = gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False, weight_initializer=mx.init.Bilinear())
        self.upsampled_flow_3_to_2 = gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False, weight_initializer=mx.init.Bilinear())

        #self.finalpredict = gluon.nn.Conv2DTranspose(3, kernel_size=8, strides=4, padding=2, weight_initializer=mx.init.Bilinear())
        #self.finalpredict.weight.lr_mult = 0.0

    def hybrid_forward(self, F, x):
        # Contracting Part
        conv_2 = self.conv2(self.conv1(x))
        conv_3 = self.conv3_1(self.conv3(conv_2))
        conv_4 = self.conv4_1(self.conv4(conv_3))
        conv_5 = self.conv5_1(self.conv5(conv_4))
        conv_6 = self.conv6(conv_5)

        # Expanding Part
        upconv5 = self.deconv5(conv_6)
        pre_flow5 = F.concat(conv_5, upconv5, dim=1)
        flow5 = self.predictflow5(pre_flow5)

        upconv4 = self.deconv4(pre_flow5)
        upsampledflow5 = self.upsampled_flow_5_to_4(flow5)
        pre_flow4 = F.concat(conv_4, upconv4, upsampledflow5, dim=1)
        flow4 = self.predictflow4(pre_flow4)

        upconv3 = self.deconv3(pre_flow4)
        upsampledflow4 = self.upsampled_flow_4_to_3(flow4)
        pre_flow3 = F.concat(conv_3, upconv3, upsampledflow4, dim=1)
        flow3 = self.predictflow3(pre_flow3)

        upconv2 = self.deconv2(pre_flow3)
        upsampledflow3 = self.upsampled_flow_3_to_2(flow3)
        pre_flow2 = F.concat(conv_2, upconv2, upsampledflow3, dim=1)
        flow2 = self.predictflow2(pre_flow2)

        return flow2, flow3, flow4, flow5


class EPError(gluon.loss.Loss):
    def __init__(self, batch_axis = 0, **kwargs):
        super(EPError, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, target):

        #loss = F.norm(pred-target, 2, axis=1, keepdims=True)
        loss = F.norm(pred-target, axis=1, keepdims=True)

        return loss.mean()


def train_target(label, target_size):
    #print(target_size)
    flow2_size = target_size
    flow3_size = (target_size[0] >> 1, target_size[1]>>1)
    flow4_size = (target_size[0] >> 2, target_size[1]>>2)
    flow5_size = (target_size[0] >> 3, target_size[1]>>3)
    flow_target2 = BilinearResize2D(label, *flow2_size)
    flow_target3 = BilinearResize2D(label, *flow3_size)
    flow_target4 = BilinearResize2D(label, *flow4_size)
    flow_target5 = BilinearResize2D(label, *flow5_size)

    return flow_target2, flow_target3, flow_target4, flow_target5
