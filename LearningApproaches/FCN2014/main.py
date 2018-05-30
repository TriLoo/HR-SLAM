import model
import mxnet as mx
from mxnet import nd
import readVOC2012
from mxnet import gluon

class_num = 21
net = model.FCNet(class_num)

input_shape = (320, 480)

'''
x = nd.random_uniform(shape=(1, 3, *input_shape)).as_in_context(mx.gpu())

y = net(x)
print(y.shape)    # output (1, 21, 320, 480)
'''

batch_size = 8

# prepare the data set
train_set = readVOC2012.VOCSegDataset(train=True, crop_size = input_shape)
test_set = readVOC2012.VOCSegDataset(False, input_shape)

# prepare the data iterator
train_data = gluon.data.DataLoader(train_set, batch_size, True, last_batch='discard')
test_data = gluon.data.DataLoader(test_set, batch_size, False, last_batch='discard')

# declare the needed loss and trainer
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':.1, 'wd':1e-3})

# the network has been initialzied during its creation
#net.initialize()

model.train(net, train_data, test_data, loss, trainer, mx.gpu(), 200, True)

try:
    print('Trying to use net.save_params() to save parameters.')
    net.save_params('FCNet.params')
except:
    print('Using net.collect_params().save() to save parameters.')
    net.collect_params().save('FCNet.params')
