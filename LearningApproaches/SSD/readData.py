import mxnet as mx
from mxnet import nd
from mxnet.ndarray.contrib import MultiBoxPrior
#from mxnet import image

data_shape = 256
batch_size = 2

rgb_mean = nd.array([123, 117, 104])

data_dir = '/home/slz/.mxnet/datasets/'
def get_iterators(data_shape, batch_size):
    class_name = ['pikachu']
    num_class = len(class_name)
    data_shape = (3, data_shape, data_shape)

    train_iter = mx.image.ImageDetIter(
        batch_size,
        data_shape,
        path_imgrec=data_dir + 'pikachu.rec',
        path_imgidx=data_dir + 'train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200
    )

    val_iter = mx.image.ImageDetIter(
        batch_size,
        data_shape,
        path_imgrec=data_dir + 'pikachu_val.rec',
        shuffle=False,
        mean=True
    )

    return train_iter, val_iter, class_name, num_class


'''
train_data, val_data, class_names, num_class = get_iterators(data_shape, batch_size)
batch = train_data.next()
print(batch)
'''




