from mxnet import gluon
from mxnet import nd
from mxnet.ndarray.contrib import MultiBoxPrior

# Several keypoints should be paied the most attentions:
#   * how to prepare the training data: rec
#   * how to define the loss function
#   * how to test the trained model

# predict the probability of each anchor
def class_predict(num_anchors, num_classes):
    return gluon.nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# predict the position of each anchor
def box_predict(num_anchors):
    return gluon.nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)

# define downsample block: downsample the dimension by half
def down_sample(num_filters):
    out = gluon.nn.HybridSequential()
    for _ in range(2):
        out.add(gluon.nn.Conv2D(num_filters, kernel_size=3, strides=1, padding=1))
        out.add(gluon.nn.BatchNorm(in_channels=num_filters))
        out.add(gluon.nn.Activation('relu'))
    out.add(gluon.nn.MaxPool2D(strides=2))

    return out

# define Merge layer: merget two array of different dimension
# flatten layer: flatten input (Batch, a, b, c,...) to a 2D array (Batch, a * b * c * ...)
# during flatten, channel prior, because we want the positions of a anchor (4 numbers) to be continuous
# now flatten order: height -> width -> channel
# After transpose & flatten: (pos{1}{1}_0, pos{1}{1}_..., pos{1}{1}_3, pos{1}{2}_0, ..., pos{1}{2}_3 ...
def flatten_predict(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()

# Concatenate different samples:
# Result: [sampA; sampB; sampC ...]
def concat_predict(preds):
    return nd.concat(*preds, dim=1)    # channel direction


# Define the main body network, replace this body using other net architecture
def body():
    net = gluon.nn.HybridSequential()
    for nfilter in [16, 32, 64]:
        net.add(down_sample(nfilter))
    return net


# define a toy model
# be aware of how to arrange the SSD elements
def toy_ssd_model(num_anchors, num_classes):
    downsamples = gluon.nn.Sequential()
    for _ in range(3):
        downsamples.add(down_sample(128))

    classpredictors = gluon.nn.Sequential()
    boxpredictors = gluon.nn.Sequential()
    for _ in range(5):
        classpredictors.add(class_predict(num_anchors, num_classes))
        boxpredictors.add(box_predict(num_anchors))

    model = gluon.nn.Sequential()
    model.add(body(), downsamples, classpredictors, boxpredictors)

    return model


# the top layer of SSD
def toy_ssd_forward(x, model, sizes, ratios, verbose=True):
    body, downsamplers, classprectors, boxpredictors = model
    anchors, class_preds, box_preds = [], [], []   # null list

    # Caution: how the data flow
    x = body(x)
    for i in range(5):
        anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))






'''
cls_pred = class_predict(5, 10)
x = nd.zeros((2, 3, 20, 20))
cls_pred.initialize()
y = cls_pred(x)

print(y.shape)
'''

net = toy_ssd_model(2, 10)
x = nd.zeros((2, 3, 256, 256))
net.initialize()
y = net(x)
print(y.shape)
print('model type = ', type(net))
