from mxnet import gluon
from mxnet import image
from mxnet import nd
import numpy as np

voc_root = '/home/slz/.mxnet/datasets/VOCdevkit/VOC2012'
fname = '/home/slz/.mxnet/datasets/VOCtrainval_11-May-2012.tar'


def readVocImages(root=voc_root, train=True):
    txt_name = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_name, 'r') as f:
        images = f.read().split()

    n = len(images)
    data, label = [None] * n, [None] * n

    for i, img_file in enumerate(images):
        data[i] = image.imread('%s/JPEGImages/%s.jpg'%(root, img_file))
        label[i] = image.imread('%s/SegmentationClass/%s.png'%(root, img_file))

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


def rand_crop(data, label, height, width):
    data, rect = image.random_crop(data, (width, height))
    label = image.fixed_crop(label, *rect)
    return data, label


def image2label(im):
    data = im.astype('int32').asnumpy()
    idx = (data[:, :, 0]*256 + data[:, :, 1])*256 + data[:, :, 2]
    return nd.array(cm2lbl[idx])


def label2image(pred):
    x = pred.astype('float32').asnumpy()
    cm = nd.array(colormap).astype('uint8')
    return nd.array(cm[x, :])


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
        data, label = readVocImages(train=train)
        data = self._filter(data)
        self.data = [normalize_image(im) for im in data]
        self.label = self._filter(label)
        print('Read ' + str(len(self.data)) + ' examples.')

    def __getitem__(self, idx):
        data, label = rand_crop(self.data[idx], self.label[idx], *self.crop_size)
        data = data.transpose((2, 0, 1))
        label = image2label(label)
        return data, label

    def __len__(self):
        return len(self.data)


