import ResNet3D
import readH5

from mxnet import gluon
import mxnet as mx
from mxnet import nd

dataset = readH5.IndianDatasets('I9-9.h5')
train_data = gluon.data.DataLoader(dataset, batch_size = 1, shuffle=True)

net = ResNet3D.ResNet3D(9)
net.collect_params().load('ResNet3D.params', ctx = mx.cpu())
#net.collect_params().reset_ctx(mx.cpu())

for i, batch in enumerate(train_data):
    data, label = batch
    pred = net(data)
    print('preds = ', nd.argmax(pred, axis=-1, keepdims=True))
    print('label = ', nd.argmax(label, axis=-1, keepdims=True))
    if i == 10:
        break
    

