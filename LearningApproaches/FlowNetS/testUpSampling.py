import mxnet as mx
from mxnet import gluon
from mxnet import nd

x = nd.ones((1, 3, 12, 16))

net = gluon.nn.Sequential()
net.add(gluon.nn.Conv2D(12, 3, 2, 1))
net.initialize()
y = net(x)
print('x.shape = ', x.shape)
print('y.shape = ', y.shape)


upconv = gluon.nn.Conv2DTranspose(channels=24, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
upconv.initialize()
z = upconv(y)
print('z.shape = ', z.shape)

'''
class testUpsample(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(testUpsample, self).__init__(**kwargs)
        self.convLayer = gluon.nn.Conv2D(24, kernel_size=3, strides=2, padding=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.convLayer(x)
        y = mx.symbol.UpSampling(data=x, scale=2, num_filter=12, sample_type='nearest')

        return y

netUp = testUpsample()
netUp.initialize()
netUp.hybridize()
z = netUp(y)
print('z.shape = ', z.shape)
'''
