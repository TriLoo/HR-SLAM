# -*- coding: utf-8 -*-

__author__ = 'smh'
__date__ = '2018.08.10'

import mxnet as mx


# mx.io.CSVIter is a function, and cannot subclass functions, use MXDataIter instead
class myCSVIter(mx.io.MXDataIter):
    def __init__(self, handle):
        super(myCSVIter, self).__init__(handle)
        self._provide_data = super().provide_data
        self._provide_label = super().provide_label

    def __iter__(self):
        return self

    def reset(self):
        super().reset()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data
    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        try:
            data_batch = super().next()
            image = data_batch.data[0]
            label = data_batch.label[0]
            return mx.io.DataBatch(image, label)
        except StopIteration:
            raise StopIteration


if __name__ == '__main__':
    train_img_iter = mx.io.CSVIter(data_csv='train_datas_tmp.csv', data_shape=(4, 640, 480),
                                   label_csv='train_labels_tmp.csv', label_shape=(640, 480),
                                   batch_size=1, dtype='float32')
    #my_iter = myCSVIter(handle=train_img_iter)
    # TODO: what is the handler to a DataIterator, so why I cannot pass the train_img_iter ?
    my_iter = myCSVIter(handle=train_img_iter)

    for databatch in my_iter:
        print(databatch.data[0].shape, databatch.label[0].shape)
    print('Passed.')
