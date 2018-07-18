import numpy as np

data = np.ones(shape=(9234, 16200))

for i in range(92):
    batch_x = data[i * 100:(i+1) * 100, :]   # return (100, 16200)
    print('shape of batch_x: ', batch_x.shape)
    break
