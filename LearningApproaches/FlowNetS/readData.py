import mxnet as mx
from mxnet import gluon
from mxnet import image
import os
from mxnet import nd


rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])

data_dir = '/home/smher/Documents/DL_Datasets/data_scene_flow'

HEIGHT = 360
WIDTH = 1240

rgb_mean = nd.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

'''
    Read all datas by the own dataset implementation + memory
'''
def readKITTIImages(root = data_dir, train=True):
    img_name = root + ('/training/train.txt' if train else '/testing/val.txt')
    print(img_name)
    with open(img_name, 'r') as f:
        image_list = f.read().split()

    n = len(image_list)   #400
    #tempdata = [None] * n
    data = [None] * (n >> 1)
    label = [None] * (n >> 1)

    for i in range(0, n-1, 2):
        label[i >> 1] = image.imread(os.path.join(os.path.split(img_name)[0], 'flow_noc', image_list[i]))
        imgA = image.imread(os.path.join(os.path.split(img_name)[0], 'image_2', image_list[i]))
        imgB = image.imread(os.path.join(os.path.split(img_name)[0], 'image_2', image_list[i+1]))
        data[i >> 1] = nd.concat(imgA, imgB, dim=-1)

    return data, label   # return W-H-C


def rand_crop(data, label, height, width):
    data, rect = image.random_crop(data, (width, height))
    label = image.fixed_crop(label, *rect)
    return data, label


def normalize_img(data):
    return (data.astype('float32')/255.0 - rgb_mean) / rgb_std


class KITTIDataset(gluon.data.Dataset):
    def __filter(self, images):
        return [im for im in images if im.shape[0] > self.crop_size[0] and im.shape[1] > self.crop_size[1]]

    def __init__(self, train, crop_size):
        self.crop_size = crop_size
        data, label = readKITTIImages(train=train)
        print('len(data) = ', len(data))
        data = self.__filter(data)
        self.data = [normalize_img(im) for im in data]
        self.label = self.__filter(label)
        print('Read ' + str(len(self.data)) + ' examples.')

    def __getitem__(self, item):
        data, label = rand_crop(self.data[item], self.label[item], *self.crop_size)
        data = data.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1)).astype('float32')
        return data, label

    def __len__(self):
        return len(self.data)



