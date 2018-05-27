from mxnet import gluon
from mxnet import nd
from mxnet import image
import mxnet as mx
import model
import os

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)

    data = nd.transpose(data, (2, 0, 1))
    return data, nd.array([label]).asscalar().astype('float32')

augs = [
    image.HorizontalFlipAug(.5),
    image.CenterCropAug((227, 227))
]

test_augs = [image.CenterCropAug((227, 227))]

train_set = gluon.data.vision.ImageFolderDataset(root='~/.mxnet/datasets/oxford102/train_datas', transform=lambda X, y: transform(X, y, augs))
test_set = gluon.data.vision.ImageFolderDataset(root='~/.mxnet/datasets/oxford102/test_datas', transform=lambda X, y: transform(X, y, augs))

lossFunc = gluon.loss.SoftmaxCrossEntropyLoss()

net = model.AlexNet()
net.initialize(init=mx.init.Xavier(), ctx=mx.gpu())

total_loss = model.train(net, train_set, test_set, lossFunc, 3, 0.001, 1000, 3, verbose=True)

#print('total loss = ', total_loss)


'''
if os.path.exists(r'AlexNet.params'):
    net.collect_params().load('AlexNet.params', ctx=mx.gpu())
    print('Readed existing weight files.')
else:
    total_loss = model.train(net, train_set, test_set, lossFunc, 32, 0.001, 1000, 32, verbose=True)
    net.collect_params().save('AlexNet.params')

test_dir = '/home/slz/.mxnet/datasets/oxford102/test_datas/9_data'

img_files = os.listdir(test_dir)
for file in img_files:
    img = os.path.join(test_dir, file)
    with open(img, 'rb') as f:
        test_img = image.imdecode(f.read())

    test_data, _ = transform(test_img, -1, test_augs)
    test_loss = model.predict(net, test_data)
'''
