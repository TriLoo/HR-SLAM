import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet import autograd
import model
import readData
from time import time
from matplotlib import pyplot as plt


'''
x = nd.ones((1, 6, 384, 512))
y = net(x)

print(y[0].shape)    # output: (1, 3, 96, 128)
print(y[1].shape)    # output: (1, 3, 48, 64)
print(y[2].shape)    # output: (1, 3, 24, 32)
print(y[3].shape)    # output: (1, 3, 12, 16)
'''

'''
x = nd.ones((1, 1, 3, 3))
y = nd.norm(x, 1)
print(y)
'''

'''
data, label = readData.readKITTIImages()
print('len(data) = ', len(data))
print('len(label) = ', len(label))

print(type(data[0]))
d = (data[199]).transpose((1, 2, 0))
print('d.shape = ', d.shape)

a, b = nd.split(d, num_outputs=2, axis=-1)

plt.imshow(a.asnumpy())
plt.show()
'''

batch_size = 1
ctx = mx.cpu()

net = model.FlowNetS()
net.initialize(init=mx.init.Xavier(), force_reinit=True)
print(net)

kittidataset = readData.KITTIDataset(True, (320, 1024))
train_data = gluon.data.DataLoader(kittidataset, batch_size, True, last_batch='discard')

loss1 = model.EPError()
loss2 = model.EPError()
loss3 = model.EPError()
loss4 = model.EPError()

#net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1, 'wd':5e-4})

weights = [0.005, 0.01, 0.02, 0.08]

A, B = 0, 0

'''
Errors:

    A. cannot find 'ord' parameter in 'nd.norm()', if the loss is resided in 'with autograd.record()', this error would appaer
    B. the weights is not initalized!
'''

for epoch in range(2):
    tic = time()
    for i, batch in enumerate(train_data):
        x = batch[0].as_in_context(ctx)
        y = batch[1].as_in_context(ctx)

        with autograd.record():
            flow2_pred, flow3_pred, flow4_pred, flow5_pred = net(x)

            with autograd.pause():
                flow2_targ, flow3_targ, flow4_targ, flow5_targ = model.train_target(y, (80, 256))

            loss = loss1(flow2_pred, flow2_targ) + loss2(flow3_pred, flow3_targ) + loss3(flow4_pred, flow4_targ) + loss4(flow5_pred, flow5_targ)

        loss.backward()
        trainer.step(batch_size)

        if (i % 20) == 0:
            print('Train: %.5f'%(loss))

    print('epoch %2d, time %.1f sec'%(epoch, time()-tic))


try:
    net.collect_params().save('FlowNetS.params')
except:
    print('net.collect_params().save(...) failed.')


'''
for data, label in train_data:
    print(label.shape)
    a, b, c, d, = model.train_target(label, (96, 128))
    print('a.shape = ', a.shape)                    # (1, 3, 96, 128)
    print('b.shape = ', b.shape)                    # (1, 3, 48, 64)
    print('c.shape = ', c.shape)                    # (1, 3, 24, 32)
    print('d.shape = ', d.shape)                    # (1, 3, 12, 16)

    break
'''

'''
for i, batch in enumerate(train_data):
    print(len(batch))
    print(type(batch))
    print(batch[0].shape)
    break
'''
'''
print(type(train_data))
for data, label in train_data:
    print('type of data = ', type(data))
    print(data.shape)
    print(label.shape)

    imgA, imgB = nd.split(data, num_outputs=2, axis=1)
    print(imgA.shape)
    imgA = (imgA[0, :, :, :]).transpose((1, 2, 0))
    print(imgA.shape)
    imgA = ((imgA * readData.rgb_std[0:3]) + readData.rgb_mean[0:3]) * 255
    imgA = imgA.astype('uint8').asnumpy()
    plt.imshow(imgA)
    break

plt.show()
'''

'''
def addtownum(a, b):
    return a + b

x = (1, 2)
print(addtownum(*x))
'''

