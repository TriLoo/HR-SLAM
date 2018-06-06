from mxnet import init
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet.ndarray.contrib import MultiBoxPrior
import model
import readData
from mxnet import metric
import time

# Train the model
ctx = mx.gpu()

# Evaluation
cls_metric = metric.Accuracy()     # used for classification, see the API tutorial for more details
box_metric = metric.MAE()          # used for box prediction, mean absolute error, see API for more explaination

data_shape = 256
batch_size = 2
train_data, val_data, class_names, num_class = readData.get_iterators(data_shape, batch_size)
train_data.reshape(label_shape=(3, 5))
train_data = val_data.sync_label_shape(train_data)     # synchronize label shape with the input iterator. To Be Sure

net = model.ToySSD(num_class)
net.initialize(init.Xavier(magnitude=2), ctx=ctx)

# Note that add the weight decay in the Trainer constructure
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1, "wd":5e-4})

cls_loss = model.FocalLoss()
box_loss = model.L1Smooth()


for epoch in range(3):
    train_data.reset()
    val_data.reset()
    cls_metric.reset()
    box_metric.reset()

    tic = time.time()
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)

        with autograd.record():
            anchors, class_preds, box_preds = net(x)
            box_target, box_mask, cls_target = model.training_targets(anchors, class_preds, y)
            loss1 = cls_loss(class_preds, cls_target)
            loss2 = box_loss(box_preds, box_target, box_mask)
            loss = loss1 + loss2
        loss.backward()
        trainer.step(batch_size)

        # update metrics
        cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
        box_metric.update([box_target], [box_preds * box_mask])
    print('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec'%(epoch, *cls_metric.get(), *box_metric.get(), time.time() - tic))



'''
# Read Data
data_shape = 256
batch_size = 2
train_data, val_data, class_names, num_class = readData.get_iterators(data_shape, batch_size)
batch = train_data.next()


net = model.ToySSD(2, True)
net.initialize()

x = batch.data[0][0:1]
print('Input shape: ', x.shape)     # 1, 3, 256, 256

anchors, class_preds, box_preds = net(x)

print('output anchros: ', anchors)     #
print('output class predictions: ', class_preds.shape)    # 1, 5444, 3
print('output box predictions: ', box_preds.shape)        # 1, 21776; 21776 = (4096 + 1024 + 256 + 64 + 4) * 4

n = 256
x = nd.random_uniform(shape=(1, 3, n, n))
# MultiBoxPrior: generate N anchors per pixel of input image
# N: see gluon tutorials
# Output: y -> (1 * 327680 * 4), 327680=256 * 256 * 5
y = MultiBoxPrior(x, sizes=[0.5, 0.25, .1], ratios=[1, 2, .5])
print('y = ', y)

boxes = y.reshape((n, n, -1, 4))
print('boxes shape = ', boxes.shape)
print(boxes[128, 128, 0, :])
'''
