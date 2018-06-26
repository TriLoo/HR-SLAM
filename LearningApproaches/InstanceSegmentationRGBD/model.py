import mxnet as mx
from mxnet import gluon
from mxnet import nd


def stnlayer(data):
    #net = gluon.nn.HybridSequential()
    return mx.sym.SpatialTransformer(data, transform_type='affine')



class ASENet(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ASENet, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


class target_loss(gluon.loss.Loss):
    def __init__(self, **kwargs):
        super(target_loss, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


def train_target():
    pass
