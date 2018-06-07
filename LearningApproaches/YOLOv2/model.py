import mxnet as mx
from mxnet import metric
from mxnet import gluon
from mxnet import image
from mxnet import nd
import numpy as np
import readData

# Input: xy: the sigmoid outputs, represent the offsets of each bounding box
def transform_center(xy):
    b, h, w, n, s = xy.shape
    # tile: repeat thw whole array multiple times
    offset_y = nd.tile(nd.arange(0, h, repeat=(w*n*1), ctx=xy.context()).reshape((1, h, w, n, 1)), (b, 1, 1, 1, 1))   # repeat b times along the batch axis
    offset_x = nd.tile(nd.arange(0, w, repeat=(n*1), ctx=xy.context()).reshape((1, 1, w, n, 1)), (b, h, 1, 1, 1)) # repeat b times  along the batch channel, and n times along axis=1

    # split: num_outputs is the number of splits
    x, y = xy.split(num_outputs=2, axis=1)
    x = (x + offset_x) / w
    y = (y + offset_y) / h
    return x, y


def transform_size(wh, anchors):
    b, h, w, n, s = wh.shape
    aw, ah = nd.tile(nd.array(anchors, ctx=wh.context()).reshape((1, 1, 1, -1, 2)), (b, h, w, 1, 1)).split(num_outputs=2, axis=-1)

    w_pred, h_pred = nd.exp(wh).split(num_outputs=2, axis=-1)
    w_out = w_pred * aw / w
    h_out = h_pred * ah / h

    return w_out, h_out


def yolo2_forward(x, num_class, anchor_scales):
    stride = num_class + 5   # predict total 'num_class + 5' numbers for each anchor
    # the 4th dimension is the number of anchors
    x = x.tranpose((0, 2, 3, 1))
    # split the last dimension to (num_class + 5) for each anchor
    # Result: (batch, m, n, stride), total m * n pixels
    x = x.reshape((0, 0, 0, -1, stride))
    # get the Pr(class | object) return [0, num_class-1]
    cls_preds = x.slice_axis(begin=0, end=num_class, axis=-1)
    # get the Pr(object)
    score_pred = x.slice_axis(begin=num_class, end=num_class+1, axis=-1)
    score = nd.sigmoid(score_pred)
    # get the bounding box offsets:
    xy_pred = x.slice_axis(begin=num_class+1, end=num_class+3, axis=-1)
    xy = nd.sigmoid(xy_pred)
    x, y = transform_center(xy)
    # get the bounding box scales
    wh_pred = x.slice_axis(begin=num_class+3, end=num_class+5, axis=-1)
    w, h = transform_size(wh_pred, anchor_scales)

    # cid is the argmax channel
    cid = nd.argmax(cls_preds, axis=-1, keepdims=True)
    # convert to corner format boxes
    half_w = w / 2
    half_h = h / 2
    left = nd.clip(x-half_w, 0, 1)
    top = nd.clip(y - half_h, 0, 1)
    right = nd.clip(x + half_w, 0, 1)
    bottom = nd.clip(y + half_h, 0, 1)
    output = nd.concat(*[cid, score, left, top, right, bottom], dim=4)  # dim=4-> concated along the 4th dimension

    return output, cls_preds, score, nd.concat(*[xy, wh_pred], dim=4)


# restore the center postions of anchors from the two points of bbox
# include the center the points, width, height
def corner2center(boxes, concat=True):
    left, top, right, bottom = boxes.split(num_outputs=4, axis=-1)
    x = (left + right) / 2
    y = (top + bottom) / 2
    width = right - left
    height = bottom - top
    if concat:
        last_dim = len(x.shape) - 1
        return nd.concat(*[x, y, width, height], dim=last_dim)    # equal to dim=-1
    return x, y, width, height


# inverse of above function
def center2corner(boxes, concat=True):
    x, y, w, h = boxes.split(num_outputs=4, axis=-1)
    w2 = w / 2
    h2 = h / 2
    left = x - w2
    top = y - h2
    right = x + w2
    bottom = y + h2

    if concat:
        last_dim = len(left.shape) - 1
        return nd.concat(*[left, top, right, bottom], dim=last_dim)

    return left, top, right, bottom


# define the training target given predictions and labels
def yolo2_target(scores, boxes, labels, anchors, ignore_label=-1, thresh=0.5):
    b, h, w, n, _ = scores.shape    # n: the number of anchors
    anchors = np.reshape(np.array(anchors), (-1, 2)) # numpy doesn't support autograde
    # define ground truth, labels = (score, top, left, right, bottom...)
    gt_boxes = nd.slice_axis(labels, begin=1, end=5, axis=-1)
    target_score = nd.zeros((b, h, w, n, 1), ctx=scores.context)
    target_id = nd.ones_like(target_score, ctx=scores.context) * ignore_label
    target_box = nd.zeros((b, h, w, n, 4), ctx=scores.context)
    sample_weight = nd.zeros((b, h, w, n, 1), ctx=scores.context)

    #for i in range(output.shape[0]):
    for i in range(b):
        # find the bestmatch for each gt
        label = labels[i].anumpy()    # labels -> (b, (n, 6))
        valid_label = label[np.where(label[:, 0] > -0.5)[0], :]
        # shuffle because multi gt could possibily match to one anchor, we keep the last match randomly
        np.random.shuffle(valid_label)
        for l in valid_label:
            gx, gy, gw, gh = (l[1] + l[3])/2, (l[4]+l[2])/2, l[3] - l[1], l[4] - l[2]
            ind_x = int(gx * w)
            ind_y = int(gy * h)
            tx = gx * w - ind_x
            ty = gy * h - ind_y
            gw = gw * w
            gh = gh * h
            # find the best match using width and height only, assuming centers are identical
            # *: element-wise multiplication
            intersect = np.minimum(anchors[:, 0], gw) * np.minimum(anchors[:, 1], gh)
            ovps = intersect / (gw * gh + anchors[:, 0] * anchors[:, 1] - intersect)
            best_match = int(np.argmax(ovps))
            target_id[b, ind_y, ind_x, best_match, :] = l[0]
            target_score[b, ind_y, ind_x, best_match, :] = 1.0
            tw = np.log(gw / anchors[best_match, 0])
            th = np.log(gh / anchors[best_match, 1])

            target_box[b, ind_y, ind_x, best_match, :] = mx.nd.array([tx, ty, tw, th])
            sample_weight[b, ind_y, ind_x, best_match, :] = 1.0

    return target_id, target_score, target_box, sample_weight


class YOLO2Output(gluon.nn.HybridSequential):
    def __init__(self, num_classes, anchor_scales, **kwargs):
        super(YOLO2Output, self).__init__(**kwargs)
        assert num_classes>0, "number of classes should > 0, given{}".format(num_classes)

        self._num_class = num_classes
        assert isinstance(anchor_scales, (list, tuple)), 'list or tuple of anchor scales required'
        assert len(anchor_scales) > 0, 'at least one anchor scale required'

        for anchor in anchor_scales:
            assert len(anchor) == 2, 'expected each anchor scale to be (width, height), provided {}'.format(anchor)

        self._anchor_scales = anchor_scales
        out_channels = len(anchor_scales) * (num_classes + 1 + 4)
        with self.name_scope():
            self.output = gluon.nn.Conv2D(out_channels, 1, 1)

    def hybrid_forward(self, F, x):
        return self.output(x)




# define own evaluation metric
# mx.metric.EvalMetric: Base class for all evaluation metrics, more details see the API tutorials
class LossRecorder(mx.metric.EvalMetric):
    def __init__(self, name):
        super(LossRecorder, self).__init__(name)

    def update(self, labels, preds=0):
        for loss in labels:
            if isinstance(loss, mx.nd.NDArray):
                loss = loss.asnumpy()

            self.sum_metric += loss.sum()
            self.num_inst += 1


# len(scales) = 2
scales = [[3.3004, 3.59034],
          [9.84923, 8.23783]]


def conv_at_layer(in_data, kernel, stride, num_filter, activation='leaky', padding=1, use_bn = True):
    out_data = mx.symbol.Convolution(in_data, kernel=kernel, stride=stride, pad=padding, num_filter=num_filter)
    # use BN before the activation
    if use_bn:
        out_data = mx.symbol.BatchNorm(out_data)
    out_data = mx.symbol.LeakyReLU(out_data, act_type=activation)

    return out_data


# Construct the blocks using Function
def conv3_4block(output_channel):
    mid_channel = output_channel // 2
    blk = gluon.nn.HybridSequential()
    blk.add(
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(mid_channel, 1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.MaxPool2D(strides=2)
    )

    return blk


def conv5_6block(output_channel):
    mid_channel = output_channel // 2
    blk = gluon.nn.HybridSequential()
    blk.add(
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(mid_channel, 1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(mid_channel, 1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
        gluon.nn.Conv2D(output_channel, 3, padding=1),
        gluon.nn.BatchNorm(),
        gluon.nn.LeakyReLU(0.1),
    )

    return blk


# reshape the 26 * 26 * 512 to 13 * 13 * 2048
def stack_neightbor(in_data, factor=2):
    # output: (batch, channel, h/2, w)
    out = mx.symbol.reshape(in_data, shape=(0, 0, -4, -1, factor, -2)) # -4后面的两个参数表明h维被分割成h/2和2(factor)了
    out = mx.symbol.transpose(out, axes=(0, 1, 3, 2, 4))
    out = mx.symbol.reshape(out, shape=(0, -3, -1, -2))
    out = mx.symbol.reshape(out, shape=(0, 0, 0, -4, -1, factor))
    out = mx.symbol.transpose(out, axes=(0, 1, 4, 2, 3))
    out = mx.symbol.reshape(out, shape=(0, -3, -1, -2))

    return out


class DarkNet19(gluon.nn.HybridBlock):
    def __init__(self, num_class, anchor_scales, **kwargs):   # for VOC: num_class = 125: 5 * (120 + 5)
        super(DarkNet19, self).__init__(**kwargs)
        self.net = gluon.nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(32, 3, padding=1),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.1),                # conv1, Set the alpha to 0.1
                gluon.nn.MaxPool2D(strides=2),          # pool1
                gluon.nn.Conv2D(64, 3, padding=1),
                gluon.nn.BatchNorm(),
                gluon.nn.LeakyReLU(0.1),                # conv2 end
                gluon.nn.MaxPool2D(strides=2),          # pool2
                conv3_4block(128),                      # conv3 + pool3
                conv3_4block(256),                      # conv4 + pool4
                conv5_6block(512),                      # conv5
            )

        self.pool5 = gluon.nn.MaxPool2D(strides=2)          # pool5
        self.conv6 = conv5_6block(1024)

        #self.output_nums = num_anchor_scales * (num_class + 1 + 4)
        #self.outputlayer = gluon.nn.Conv2D(self.output_nums, kernel_size=1)
        self.outputlayer = YOLO2Output(num_class, anchor_scales)


        self.avgpool = gluon.nn.GlobalAvgPool2D()

        self.passthrough = self.net[-1]

        # used for detection
        self.conv8 = gluon.nn.HybridSequential()
        self.conv8.add(
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1),
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1),
        )

        self.conv9 = gluon.nn.HybridSequential()
        self.conv9.add(
            gluon.nn.Conv2D(1024, 3, padding=1),
            gluon.nn.BatchNorm(),
            gluon.nn.LeakyReLU(0.1)
        )


    #def hybrid_forward(self, F, x, detection = True,  *args, **kwargs):
    def hybrid_forward(self, F, x,  *args, **kwargs):
        mid_x = self.net(x)
        #if detection:
        mid_y = self.pool5(mid_x)
        mid_y = self.conv6(mid_y)
        mid_y = self.conv8(mid_y)
        mid_y = mx.symbol.concat(mid_y, stack_neightbor(mid_x), dim=1)
        mid_y = self.conv9(mid_y)
        y = self.outputlayer(mid_y)

        return mx.symbol.softmax(y)


def process_image(fname, net, data_shape, ctx=mx.gpu()):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())

    data = image.imresize(im, data_shape, data_shape)
    data = (data.astype('float32') - readData.rgb_mean) / readData.rgb_std

    return data.transpose((2, 0, 1)).expand_dims(axis=0), im


def predict(x, net):
    x = net(x)
    output, cls_prob, xywh = yolo2_forward(x, 2, scales)

    return nd.contrib.box_nms(output.reshape((0, -1, 6)))




