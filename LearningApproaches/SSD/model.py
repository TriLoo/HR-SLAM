import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.ndarray.contrib import MultiBoxPrior
from mxnet.ndarray.contrib import MultiBoxTarget
from mxnet import image
import readData
from mxnet.ndarray.contrib import MultiBoxDetection
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

colors = ['blue', 'green', 'red', 'black', 'magenta']
class_name = ['pikachu']

# Several keypoints should be paied the most attentions:
#   * how to prepare the training data: rec
#   * how to define the loss function
#   * how to test the trained model

# predict the probability of each anchor
def class_predict(num_anchors, num_classes):
    return gluon.nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)    # output: (num_classes+1) per anchor, total 'num_anchors' anchors

# predict the position of each anchor
def box_predict(num_anchors):
    return gluon.nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)    # each anchor has 4 output channel for each axis of one anchor

# define downsample block: downsample the dimension by half
def down_sample(num_filters):
    out = gluon.nn.HybridSequential()
    for _ in range(2):
        out.add(gluon.nn.Conv2D(num_filters, kernel_size=3, strides=1, padding=1))
        out.add(gluon.nn.BatchNorm(in_channels=num_filters))
        out.add(gluon.nn.Activation('relu'))
    out.add(gluon.nn.MaxPool2D(strides=2))

    return out

# Very Important
# define Merge layer: merget two array of different dimension
# flatten layer: flatten input (Batch, a, b, c,...) to a 2D array (Batch, a * b * c * ...)
# during flatten, channel prior, because we want the positions of a anchor (4 numbers) to be continuous, the channel represent the class prediction of ONE anchor
# now flatten order: height -> width -> channel
# After transpose & flatten: (pos{1}{1}_0, pos{1}{1}_1, ..., pos{1}{1}_3, pos{1}{2}_0, ..., pos{1}{2}_3 ...
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
    body, downsamplers, classpredictors, boxpredictors = model
    anchors, class_preds, box_preds = [], [], []   # null list

    # Caution: how the data flow
    x = body(x)
    for i in range(5):
        anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))     # generate 4 box per pixel of x, total 5 scales
        class_preds.append(flatten_predict(classpredictors[i](x)))
        box_preds.append(flatten_predict(boxpredictors[i](x)))
        if verbose:
            print('Predict scale', i, x.shape, 'with', anchors[-1].shape[1], 'achors')

        if i < 3:
            x = downsamplers[i](x)
        elif i == 3:
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(x.shape[2], x.shape[3]))   # Global kernel

    return (concat_predict(anchors),      # the channel dim is the 
            concat_predict(class_preds),
            concat_predict(box_preds))


class ToySSD(gluon.nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ToySSD, self).__init__(**kwargs)

        # Total 5 scale, 4 anchors per scale
        self.sizes = [[.2, .272], [.37, .447], [.54, .619],
                      [.71, .79], [.88, .961]]
        self.ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes

        self.verbose = verbose
        num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1   # = 4
        with self.name_scope():
            self.model = toy_ssd_model(num_anchors, num_classes)

    def forward(self, x):
        anchors, class_preds, box_preds = toy_ssd_forward(x, self.model, self.sizes, self.ratios, self.verbose)
        class_preds = class_preds.reshape(shape=(0, -1, self.num_classes+1))
        return anchors, class_preds, box_preds


def training_targets(anchors, class_preds, labels):
    class_preds = class_preds.transpose(axes=(0, 2, 1))
    # compute multibox training targets
    return MultiBoxTarget(anchors, labels, class_preds)


# class prediction loss
class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        # Pick: pick elements from an input array according to the input indices along the given axis
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = - self._alpha * (1 - pj)**self._gamma * pj.log()
        return loss.mean(axis = self._batch_axis, exclude=True)


# bounding box prediction loss
class L1Smooth(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(L1Smooth, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar = 1.0)
        # use exclude to perform reduction on axis that are not in given axis
        return loss.mean(self._batch_axis, exclude=True)


def process_image(fname, data_shape):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())

    data = image.imresize(im, data_shape, data_shape)
    data = data.astype('float32') - readData.rgb_mean

    return data.transpose((2, 0, 1)).expand_dims(axis=0), im

def predict(x, net, ctx = mx.gpu()):
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    cls_probs = nd.SoftmaxActivation(cls_preds.transpose((0, 2, 1)), model='channel')

    return MultiBoxDetection(cls_probs, box_preds, anchors, force_suppress=True, clip=False)


def box_to_rect(box, color, linewidth=3):
    box = box.asnumpy()
    return plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=color, linewidth=linewidth)


mpl.rcParams['figure.figsize'] = (6, 6)
def display(im, out, threshold=0.5):
    plt.imshow(im.asnumpy())
    for row in out:
        row = row.asnumpy()
        class_id, score = int(row[0]), row[1]
        if class_id < 0 or score < threshold:
            continue
        color = colors[class_id % len(colors)]
        box = row[2:6] * np.array([im.shape[0], im.shape[1]] * 2)
        rect = box_to_rect(nd.array(box), color, 2)
        plt.gca().add_patch(rect)

        text = class_name[class_id]
        plt.gca().text(box[0], box[1], '{:2} {:.2f}'.format(text, score), bbox=dict(facecolor=color, alpha=0.5), fontsize=10, color='white')
    plt.show()


'''
cls_pred = class_predict(5, 10)
x = nd.zeros((2, 3, 20, 20))
cls_pred.initialize()
y = cls_pred(x)

print(y.shape)

net = toy_ssd_model(2, 10)
x = nd.zeros((2, 3, 256, 256))
net.initialize()
y = net(x)
print(y.shape)
print('model type = ', type(net))
'''
