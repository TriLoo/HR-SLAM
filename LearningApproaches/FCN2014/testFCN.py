import mxnet as mx
import model
import readVOC2012
from time import time
from mxnet import nd
#import pyplot as plt
import matplotlib.pyplot as plt

class_num = 21
net = model.FCNet(class_num)

net.collect_params().load('FCNet.params', ctx=mx.gpu())     # the *params is obtained from GPU, so here the ctx must be set to mx.gpu() too, otherwise, error

#net.collect_params().reset_ctx(mx.gpu())

test_images, test_labels = readVOC2012.readVocImages(train=False)

img = test_images[0].astype('float32')
label = test_labels[0].astype('float32')
print('Input shape = ', img.shape)

input_shape = (320, 480)
img, label = readVOC2012.rand_crop(img, label, *input_shape)


st = time()

y = model.predict(net, img, mx.gpu(), readVOC2012.normalize_image)

print('time = ', time() - st)

#print(type(y))
#print(y.shape)

imgFinal = readVOC2012.label2image(y)

print('Done')

print(type(imgFinal))
print(imgFinal.shape)


plt.imshow(imgFinal.asnumpy())
plt.show()

plt.figure
plt.imshow(label.asnumpy())
plt.show()

