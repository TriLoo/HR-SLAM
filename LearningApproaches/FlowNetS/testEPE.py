import model
from mxnet import nd

a = nd.random_uniform(0, 1, shape=(1, 3, 10, 32))
b = nd.random_uniform(0, 1, shape=(1, 3, 10, 32))

loss = model.EPError()
c = loss(a, b)
print(c)



