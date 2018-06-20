import mxnet as mx
from mxnet import gluon
from mxnet import image
import os
from mxnet import nd


rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])

data_dir = '/home/smher/Documents/DL_Datasets/data_scene_flow'

HEIGHT = 375
WIDTH = 1242

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
        label[i >> 1] = image.imread(os.path.join(os.path.split(img_name)[0], 'flow_noc', image_list[i])).transpose((2, 0, 1))
        imgA = image.imread(os.path.join(os.path.split(img_name)[0], 'image_2', image_list[i])).transpose((2, 0, 1))
        imgB = image.imread(os.path.join(os.path.split(img_name)[0], 'image_2', image_list[i+1])).transpose((2, 0, 1))
        data[i >> 1] = nd.concat(imgA, imgB, dim=0)

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



