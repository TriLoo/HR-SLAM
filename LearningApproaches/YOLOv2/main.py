import mxnet as mx
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet import autograd
import model
import readData
from time import time


data_shape = 256
batch_size = 2
train_data, test_data, class_names, num_class = readData.get_iterators(data_shape, batch_size)

# A very good blog: https://zhuanlan.zhihu.com/p/36902889
#batch = train_data.next()
# label: class_id, left, top, right, bottom, normlized to 0 - 1
#print('batch=', batch)    # = data (2, 3, 256, 256), label (2, 1, 5)
#print('batch.data = ', batch.data, len(batch.data))     # = 2 * 3 * 256 * 256 Ndarray, 1
#print('batch.data[0] = ', batch.data[0])                 # = 2 * 3 * 256 * 256

net = model.DarkNet19(2, model.scales)   # the class = 2, including the background as dummy!

ctx = mx.gpu()
net.initialize(init=mx.init.Xavier(), ctx=mx.gpu())
#net.initialize(init=mx.init.Xavier())
net.hybridize()


#ctx = mx.gpu()
#ctx = mx.cpu()
#net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':1, 'wd':5e-4})


# define loss function
sec_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
l1_loss = gluon.loss.L1Loss()

obj_loss = model.LossRecorder('objectness_loss')
cls_loss = model.LossRecorder('classification_loss')
box_loss = model.LossRecorder('box_refine_loss')

positive_weight = 5.0
negative_weight = .1
class_weight=1.0
box_weight=5.0

batch = train_data.next()
print(batch)
for epoch in range(2):
    train_data.reset()
    cls_loss.reset()
    obj_loss.reset()
    box_loss.reset()

    tic = time()
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            x = net(x)
            #print('x.shape = ', x.shape)   # 2 * 14 * 8 * 8
            # xywh: center, width, height
            output, cls_pred, score, xywh = model.yolo2_forward(x, 2, model.scales)
            #print('score.shape = ', score.shape)
            #print('After Forward.')
            with autograd.pause():     # return a scope context to be used in 'with' statement for codes that do not need gradients to be calculated
                tid, tscore, tbox, sample_weight = model.yolo2_target(score, xywh, y, model.scales, thresh=.5)   # score:1 * batch_size
                #print('After Forward.')

            loss1 = sec_loss(cls_pred, tid, sample_weight * class_weight)
            score_weight = nd.where(sample_weight>0, nd.ones_like(sample_weight) * positive_weight, nd.ones_like(sample_weight) * negative_weight)
            loss2 = l1_loss(score, tscore, score_weight)
            loss3 = l1_loss(xywh, tbox, sample_weight * box_weight)
            loss = loss1 + loss2 + loss3

        loss.backward()
        trainer.step(batch_size)

        # update metrics
        cls_loss.update(loss1)
        obj_loss.update(loss2)
        box_loss.update(loss3)

    print('epoch %2d, train %s %.5f, %s %.5f, %s %.5f, time %.1f sec'%(epoch, *cls_loss.get(), *obj_loss.get(), *box_loss.get(), time() - tic))

try:
    net.save_params('SSD.params')
except:
    print('First try failed. Try second time...')
    net.collect_params().save('SSD.params')
    print('Parameters have been saved to SSD.params')

'''
def test_stack_neightbor(in_data, factor=2):                    # in_data: 1, 3, 416, 416
    out = mx.sym.reshape(in_data, shape=(0, 0, -4, -1, factor, -2)) # -4后面的两个参数表明h维被分割成h/2和2(factor)了
    #print('out shape = ', out.shape)                           # 1, 3, 208, 2, 416
    out = mx.symbol.transpose(out, axes=(0, 1, 3, 2, 4))
    #print('out shape = ', out.shape)                           # 1, 3, 2, 208, 416
    out = mx.sym.reshape(out, shape=(0, -3, -1, -2))
    #print('out shape = ', out.shape)                           # 1, 6, 208, 416
    out = mx.sym.reshape(out, shape=(0, 0, 0, -4, -1, factor))
    #print('out shape = ', out.shape)                           # 1, 6, 208, 208, 2
    out = mx.sym.transpose(out, axes=(0, 1, 4, 2, 3))
    #print('out shape = ', out.shape)                           # 1, 6, 2, 208, 208
    out = mx.sym.reshape(out, shape=(0, -3, -1, -2))                # output: 1, 12, 208, 208

    return out


x = nd.random_uniform(shape=(1, 3, 416, 416))
#x = mx.symbol.var('data')

print('x type = ', type(x))
#y = model.stack_neightbor(x)
#y = test_stack_neightbor(x)

# test the net
start = time()
y = net(x)   # correct, y = (1, 125, 13, 13)
print('time = ', time() - start)   # hybridize: 0.03508


print('y type = ', type(y))
print('y shape = ', y.shape)
'''
