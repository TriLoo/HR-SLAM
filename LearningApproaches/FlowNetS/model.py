import mxnet as mx
from mxnet import gluon
import numpy as np

class Contract(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(Contract, self).__init__(**kwargs)
        self.conv1 = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), strides=(2, 2), padding=1, activation='relu')
        self.conv2 = gluon.nn.Conv2D(channels=128, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu')
        self.conv3 = gluon.nn.Conv2D(channels=256, kernel_size=(5, 5), strides=(2, 2), padding=1, activation='relu')
        self.conv4 = gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv5 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(2, 2), padding=1, activation='relu')
        self.conv6 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv7 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=2, padding=1, activation='relu')
        self.conv8 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv9 = gluon.nn.Conv2D(channels=1024, kernel_size=(3 ,3), strides=1, padding=1, activation='relu')

        self.netContract = gluon.nn.Sequential()
        self.netContract.add(self.conv1,
                             self.conv2,
                             self.conv3,
                             self.conv4,
                             self.conv5,
                             self.conv6,
                             self.conv7,
                             self.conv8,
                             self.conv9)

    def forward(self, x):
        return self.net(x)


# define the 'upconvolutional' layer
class UpConv(gluon.nn.Block):
    def __init__(self, output_channels, **kwargs):
        super(UpConv, self).__init__(**kwargs)

        #
        self.Deconv = gluon.nn.Conv2DTranspose(channels=output_channels, kernel_size=(), strides=())

    def forward(self, x):
        return self.Deconv(x)


# define Refinement block in FlowNetS
class Refinement(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(Refinement, self).__init__(**kwargs)

        self.RefineNet = UpConv()

    # 输入 x 为 Contract 部分的输出
    def forward(self, x):
        return self.RefineNet(x)

# 由于需要结合来自底层部分的结果，所以这里就直接写在一起吧，不然不好传参数啊
class FlowNetS(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(FlowNetS, self).__init__(**kwargs)
        self.conv1 = gluon.nn.Conv2D(channels=64, kernel_size=(7, 7), strides=(2, 2), padding=3, activation='relu')
        self.conv2 = gluon.nn.Conv2D(channels=128, kernel_size=(5, 5), strides=(2, 2), padding=2, activation='relu')
        self.conv3 = gluon.nn.Conv2D(channels=256, kernel_size=(5, 5), strides=(2, 2), padding=2, activation='relu')
        self.conv3_1 = gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv4 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(2, 2), padding=1, activation='relu')
        self.conv4_1 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv5 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=2, padding=1, activation='relu')
        self.conv5_1 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=1, activation='relu')
        self.conv6 = gluon.nn.Conv2D(channels=1024, kernel_size=(3, 3), strides=1, padding=1, activation='relu')

        self.deconv5 = UpConv(output_channels=2)

    def forward(self, x):
        # Contracting Part
        conv_2 = self.conv2(self.conv1(x))
        conv_3 = self.conv3_1(self.conv3(conv_2))
        conv_4 = self.conv4_1(self.conv4(conv_3))
        conv_5 = self.conv5_1(self.conv5(conv_4))
        conv_6 = self.conv6(conv_5)

        # Expanding Part, Important
        upconv5 =




