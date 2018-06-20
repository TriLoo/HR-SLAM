import mxnet as mx
from mxnet import gluon
from mxnet import image
import os
from mxnet import nd


rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])

data_dir = '/home/slz/.mxnet/datasets/'


def get_iterators(data_shape, batch_size):
    class_name = ['pikachu', 'dummy']
    num_class = len(class_name)
    data_shape = (3, data_shape, data_shape)

    train_iter = mx.image.ImageDetIter(
        batch_size,
        data_shape,
        path_imgrec=data_dir + 'pikachu.rec',
        path_imgidx=data_dir + 'train.idx',
        shuffle=True,
        mean=True,
        std=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200
    )

    val_iter = mx.image.ImageDetIter(
        batch_size,
        data_shape,
        path_imgrec=data_dir + 'pikachu_val.rec',
        shuffle=False,
        mean=True,
        std=True
    )

    return train_iter, val_iter, class_name, num_class


'''
train_data, val_data, class_names, num_class = get_iterators(data_shape, batch_size)
batch = train_data.next()
print(batch)
'''




