import OpenMaxLayer
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
import ResNet
import numpy as np
import cv2

mcv = np.array([1, 1])
query_score = np.array([2, 2])

# calc_distance func
#res = model.calc_distance(query_score, mcv, 0.1)
#print(res)


'''
net = ResNet.finetune_resnet18
net.collect_params().load('resnet.params', ctx=mx.cpu())
#print(net.features)
#print(net.output)

img = cv2.imread('19_1.jpg')     # numpy.ndarray

img = nd.array(img)

augs = transforms.Compose(
        [transforms.RandomResizedCrop((420, 312), scale=(0.8, 1.0)),
        transforms.ToTensor()])

for aug in augs:
    img = aug(img)

img = img * 255
img = img.expand_dims(axis=0)    # insert a new axis of size 1 into the array shape
print(img.shape)

preds = net(img)     # the output's shape is (1, 100)
print(preds.shape)
print('class = ', preds.argmax(axis=1))
#print(preds)
print('sum of preds = ', nd.sum(preds))
'''

