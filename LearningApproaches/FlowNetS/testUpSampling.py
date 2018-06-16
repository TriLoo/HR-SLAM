import mxnet as mx
from mxnet import gluon
from mxnet import nd
import numpy as np

x = nd.ones((1, 3, 12, 16))

net = gluon.nn.Sequential()
net.add(gluon.nn.Conv2D(12, 3, 2, 1))
net.initialize()
y = net(x)
print('x.shape = ', x.shape)
print('y.shape = ', y.shape)

'''
upconv = gluon.nn.Conv2DTranspose(channels=24, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
upconv.initialize()
z = upconv(y)
print('z.shape = ', z.shape)
'''
scale = 2
class testUpsample(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(testUpsample, self).__init__(**kwargs)
        self.convLayer = gluon.nn.Conv2D(24, kernel_size=3, strides=2, padding=1)
        self.transLayer = gluon.nn.Conv2DTranspose(12, 4, 2, 1, weight_initializer=mx.init.Bilinear())
        #self.transLayer.collect_params().set_lr_scale(0.0)
        self.transLayer.weight.lr_mult = 0.0
        self.xx1 = mx.init.Bilinear()


    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.convLayer(x)
        y = self.transLayer(x)
        #y = mx.symbol.UpSampling(x, self.xx1, scale=2, num_filter=12, sample_type='bilinear', num_args=2)

        return y

netUp = testUpsample()
netUp.initialize()
netUp.hybridize()
z = netUp(y)
print('z.shape = ', z.shape)


'''
xx = nd.random_normal(shape=[1,1,256,256],ctx=mx.cpu())
xx1 = nd.random_normal(shape=[1,1,4,4],ctx=mx.cpu())
temp = nd.UpSampling(xx,xx1, num_filter=1, scale=2, sample_type='bilinear', num_args=2)
print(temp.shape)
'''


