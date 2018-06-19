import mxnet as mx
from mxnet import gluon
from mxnet import image
import os
from mxnet import nd


rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])

data_dir = '~/Documents/DL_datasets/data_scene_flow'

'''
    Read all datas by the own dataset implementation + memory
'''

def readKITTIImages(root = data_dir, train=True):
    img_name = root + ('/training/train.txt' if train else '/testing/val.txt')
    with open(img_name, 'r') as f:
        image_list = f.read().split()

    n = len(image_list)
    data = [None] * n
    label = [None] * n

    for i, img_file in enumerate(image_list):
        if i % 2 == 0:
            imgA = image.imread(os.path.join(os.path.split(img_name)[0], 'image_2/', 'img_file'))    #
            data[i] = imgA
            label[i] = image.imread(os.path.join(os.path.split(img_name)[0], 'flow_noc/', 'img_file'))
        else:
            continue

    return data, label


class KITTIDataset(gluon.data.Dataset):
    def __init__(self, train, crop_size):
        self.crop_size = crop_size
        data, label = readKITTIImages()
        self.data = data
        self.label = label

    def __getitem__(self, item):
        data = self.data[item].transpose((2, 0, 1))
        return data

    def __len__(self):
        return len(self.data)



