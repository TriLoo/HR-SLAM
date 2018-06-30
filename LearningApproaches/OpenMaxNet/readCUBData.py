from mxnet import nd
from mxnet.gluon.data.vision import transforms
from mxnet import gluon
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


# generate dataset from image list file
batch_size = 2
data_shape = 256

rgb_mean = nd.array([123, 117, 104])

data_dir = '/home/slz/.mxnet/datasets/CUB100'

augs = transforms.Compose([
    # Notes: the input size of randomresizedcrop is (W, H)
    transforms.RandomResizedCrop((420, 312), scale=(0.8, 1.0)),      # input data shape: H, W, C
    transforms.RandomFlipLeftRight()])

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, axes=(2, 0, 1))

    return (data, nd.array([label]).asscalar().astype('float32'))


# Work pass
#train_set = gluon.data.vision.ImageFolderDataset(data_dir, transform=lambda X, y:transform(X, y, augs))
#data = gluon.data.DataLoader(train_set, batch_size = 2, shuffle=True)

'''
for X, label in data:
    print(X.shape)
    X = X.transpose((0, 2, 3, 1)).clip(0, 255) / 255.0
    print(X.shape)
    img = X[1]
    print(X.shape)
    print(label)
    plt.imshow(img.asnumpy())
    plt.show()
    break
'''


'''
x = nd.random_uniform(0, 1, shape=(332, 500, 3))
label = 1
print('x.shape = ', x.shape)
y, l = transform(x, label, augs)
print('y.shape = ', y.shape)
print('label.shape = ', l.shape)
'''






