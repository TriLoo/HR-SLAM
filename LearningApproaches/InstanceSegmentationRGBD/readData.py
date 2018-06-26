import mxnet as mx
from mxnet import nd
import scipy.io as scio
import h5py

'''
Dataset:  
    NYU Dataset V2
Method:
    Input: .mat file
    Output: Data set of mxnet
    Tool: scipy.io
'''

mat_dir = '/home/smher/Documents/DL_Datasets/nyu_depth_v2_labeled.mat'

with h5py.File(mat_dir, 'r') as f:
    print('f.keys = ', f.keys())
    print('type of f = ', type(f))
    print(type(f.get('depths')))

#data = scio.loadmat(mat_dir)
#print(type(data['depths']))
#print(type(data['images']))





