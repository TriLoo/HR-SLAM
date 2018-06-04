import mxnet as mx
from mxnet import nd
import model

net = model.DarkNet19(125)
net.initialize(init=mx.init.Xavier())
net.hybridize()

def test_stack_neightbor(in_data, factor=2):                    # in_data: 1, 3, 416, 416
    out = nd.reshape(in_data, shape=(0, 0, -4, -1, factor, -2)) # -4后面的两个参数表明h维被分割成h/2和2(factor)了
    #print('out shape = ', out.shape)                           # 1, 3, 208, 2, 416
    out = nd.transpose(out, axes=(0, 1, 3, 2, 4))
    #print('out shape = ', out.shape)                           # 1, 3, 2, 208, 416
    out = nd.reshape(out, shape=(0, -3, -1, -2))
    #print('out shape = ', out.shape)                           # 1, 6, 208, 416
    out = nd.reshape(out, shape=(0, 0, 0, -4, -1, factor))
    #print('out shape = ', out.shape)                           # 1, 6, 208, 208, 2
    out = nd.transpose(out, axes=(0, 1, 4, 2, 3))
    #print('out shape = ', out.shape)                           # 1, 6, 2, 208, 208
    out = nd.reshape(out, shape=(0, -3, -1, -2))                # output: 1, 12, 208, 208

    return out


x = nd.random_uniform(shape=(1, 3, 416, 416))
#x = mx.symbol.var('data')

print('x type = ', type(x))
#y = model.stack_neightbor(x)
#y = test_stack_neightbor(x)

# test the net
y = net(x)   # correct, y = (1, 125, 13, 13)

print('y type = ', type(y))

print('y = ', y)
print('y shape = ', y.shape)

