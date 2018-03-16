import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd

# TODO: this layer is used to calculate the 1-vs-rest Layer of DOC
# DOC: Deep Open Classification of Text Documents
class OneVsRest(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(OneVsRest, self).__init__(**kwargs)
        with self.name_scope():
            self.net = gluon.nn.Sequential()
            self.net.add(
                gluon.nn.Conv2D(channels=32, kernel_size=(3, 3))
            )

            # calculating the num_class Sigmoid functions
            # for i in range(1, num_class):

    # TODO: implement the num_class's Sigmoid
    def forward(self, x, num_class):
        for i in range(1, num_class):
            nd.softmax(x)

        return nd.relu(self.net(x))

class OCN(gluon.nn.Block):
    def __init__(self, num_class, **kwargs):
        super(OCN, self).__init__(**kwargs)
        with self.name_scope():
            self.net = gluon.nn.Sequential()
            self.net.add(
                gluon.nn.Conv2D(),
                OneVsRest()
            )

    def forward(self, x):
        return nd.Softmax(self.net(x))

def lossCal(x, label):
    pre = nd.softmax(x)
    return 0 if pre == label else 1


net = OCN(10)
