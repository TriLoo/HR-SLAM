import mxnet as mx
from mxnet import gluon
import numpy as np
from mxnet import autograd
from mxnet import nd

from mxnet import image

from mxnet.gluon.model_zoo import vision as models

import utils

# load the params on GPU(0)
#pretrained_net = models.resnet18_v2(pretrained=True)
pretrained_net = models.resnet18_v2(ctx=mx.gpu(), pretrained=True)
#pretrained_net = models.resnet18_v2(ctx=mx.cpu(), pretrained=True)
print(pretrained_net)

net = gluon.nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

input_shape = (320, 480)
x = nd.random_uniform(shape=(1, 3, *input_shape)).as_in_context(mx.gpu())
y = net(x)

print('Input: ', x.shape)
print('output: ', y.shape)

classes = 21

with net.name_scope():
    net.add(
        gluon.nn.Conv2D(classes, kernel_size=1),
        gluon.nn.Conv2DTranspose(classes, kernel_size=64, padding=16, strides=32)
    )


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]    # returen two one-dimension vector, each including [0, ..., kernel_size-1]

    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center)/factor)
    weight = np.zeros(shape=(in_channels, out_channels, kernel_size, kernel_size), dtype='float32')

    weight[range(in_channels), range(out_channels), :, :] = filt

    return nd.array(weight)

conv_trans = net[-1]
conv_trans.initialize(init = mx.init.Zero(), ctx=mx.gpu())

net[-2].initialize(init=mx.init.Xavier(), ctx=mx.gpu())

batch_size = 8
x = nd.zeros((batch_size, 3, *input_shape)).as_in_context(mx.gpu())
net(x)

shape = conv_trans.weight.data().shape
conv_trans.weight.set_data(bilinear_kernel(*shape[0:3]))

loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)   # axis = 1:
ctx = mx.gpu()

net.collect_params().reset_ctx(ctx)


trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':.1, 'wd':1e-3})

def train(net, train_set, lossFunc,  batchSize, lr, epochs, period, eps = 1e-6, verbose=False):
    assert batchSize >= period and period % batchSize == 0
    #trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate':lr, 'beta1':0.9, 'beta2':0.999})
    #trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'momentum':0.9})
    total_loss = []

    train_data = gluon.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)

    for epoch in range(epochs):
        for batch_i, (data, label) in enumerate(train_data):
            #print('label = ',label)
            data = data.as_in_context(mx.gpu())
            label = label.as_in_context(mx.gpu())
            with autograd.record():
                output = net(data)
                #print('output = ', output)
                loss = lossFunc(output, label)
            loss.backward()
            trainer.step(batchSize)

            if batch_i * batchSize % period == 0:
                total_loss.append(np.mean(lossFunc(net(data).as_in_context(mx.cpu()), label.as_in_context(mx.cpu())).asnumpy()))

        if verbose:
            print('Batch Size: %d, learning rate: %f, epoch: %f, loss %.4e'%(batchSize, trainer.learning_rate, epoch, total_loss[-1]))

        if epoch + 1 % 30 == 0:
            lr = lr / 10
            trainer.set_learning_rate(lr)

        if lr < 0.000001:
            lr = 0.000001

        #if len(total_loss) > 2 and total_loss[-2] - total_loss[-1] < eps:
        #break

    return total_loss



# download VOC2012
data_root = '/home/slz/.mxnet/datasets'
voc_root = data_root + '/VOCdevkit/VOC2012'
fname = '/home/slz/.mxnet/datasets/VOCtrainval_11-May-2012.tar'
'''
#url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012'
       #'/VOCtrainval_11-May-2012.tar')
sha1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'

#fname = gluon.utils.download(url, data_root, sha1_hash=sha1)

print('fname = ', fname)
print('data root = ', data_root)

if not os.path.isfile(voc_root + 'ImageSets/Segmentation/train.txt'):
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_root)
'''

def read_images(root = voc_root, train=True):
    txt_name = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_name, 'r') as f:
        images = f.read().split()

    n = len(images)
    data, label = [None] * n, [None] * n
    for i, img_file in enumerate(images):
        data[i] = image.imread('%s/JPEGImages/%s.jpg'%(root, img_file))
        label[i] = image.imread('%s/SegmentationClass/%s.png'%(root, img_file))

    return data, label


train_images, train_labels = read_images()
img = train_images[0]
print(img.shape) # print (281, 500, 3)
print(type(img)) # NDArray

#im = Image.fromarray(img.asnumpy())
#im.show()  # Cannot open display

# To gaurantee the correspondance between train image and its label, use crop
def rand_crop(data, label, height, width):
    data, rect = image.random_crop(data, (width, height))
    label = image.fixed_crop(label, *rect)
    return data, label


# Declare the class-specified color, total 21 classes
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep',
           'sofa', 'train', 'tv/monitor']

# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0]*256 + cm[1])*256 + cm[2]] = i

def image2label(im):
    data = im.astype('int32').asnumpy()
    idx = (data[:, :, 0]*256 + data[:, :, 1])*256 + data[:, :, 2]
    return nd.array(cm2lbl[idx])

#y = image2label(train_labels[0])

#print(y[105:115, 130:140])

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def normalize_image(data):
    return (data.astype('float32')/255.0 - rgb_mean) / rgb_std

# Important, inherit from gluon.data.Dataset, to enable the enumerate operation of dataset!
# must overwrite the __getitem__, __len__
class VOCSegDataset(gluon.data.Dataset):
    def _filter(self, images):
        return [im for im in images if (im.shape[0] >= self.crop_size[0] and im.shape[1] >= self.crop_size[1])]

    def __init__(self, train, crop_size):
        self.crop_size = crop_size
        data, label = read_images(train=train)
        data = self._filter(data)
        self.data = [normalize_image(im) for im in data]
        self.label = self._filter(label)
        print('Read ' + str(len(self.data)) + ' examples.')

    def __getitem__(self, idx):
        data, label = rand_crop(self.data, self.label, *self.crop_size)
        data = data.transpose((2, 0, 1))
        label = image2label(label)
        return data, label

    def __len__(self):
        return len(self.data)


input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape)   # Read 1114
voc_test = VOCSegDataset(False, input_shape)   # Read 1078

# Use the gluon.data.DataLoader to get the iterator
# discard the last batch if batch_size does not evenly divide len(dataset): len(dataset) % batch_size != 0
train_data = gluon.data.DataLoader(voc_train, batch_size=batch_size, shuffle=True, last_batch='discard')
test_data = gluon.data.DataLoader(voc_test, batch_size=batch_size, shuffle=False, last_batch='discard')

#train(net, train_data, loss, batch_size, 0.1, 10, batch_size, verbose=True)

utils.train(train_data, test_data, net, loss, trainer, mx.gpu(), 3)

def predict(img):
    data = normalize_image(img)
    data = data.transpose((2, 0, 1)).expand_dims(axis=0)
    yhat = net(data.as_in_context(mx.gpu()))
    pred = nd.argmax(yhat, axis = 1)
    return pred.reshape((pred.shape[1], pred.shape[2]))

def label2image(pred):
    x = pred.astype('int32').asnumpy()
    cm = nd.array(colormap).astype('uint8')
    return nd.array(cm[x, :])
