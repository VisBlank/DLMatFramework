

import numpy as np
import scipy.io
from cs231n.layers import *
from cs231n.fast_layers import *


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# Check the training-time forward pass by checking means and variances
# of features both before and after spatial batch normalization
N, C, H, W = 2, 3, 4, 5

# Get data from matlab file (Helps debug by using the same data)
# x = 4 * np.random.randn(N, C, H, W) + 10
dictMat = scipy.io.loadmat('../../../test/layers/spatial_batchnorm_forward.mat')
x = dictMat['x']

print ('Before spatial batch normalization:')
print ('  Shape: ', x.shape)
print ('  Means: ', x.mean(axis=(0, 2, 3)))
print ('  Stds: ', x.std(axis=(0, 2, 3)))

# Means should be close to zero and stds close to one
gamma, beta = np.ones(C), np.zeros(C)
bn_param = {'mode': 'train'}
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print ('After spatial batch normalization:')
print ('  Shape: ', out.shape)
print ('  Means: ', out.mean(axis=(0, 2, 3)))
print ('  Stds: ', out.std(axis=(0, 2, 3)))

# Means should be close to beta and stds close to gamma
gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print ('After spatial batch normalization (nontrivial gamma, beta):')
print ('  Shape: ', out.shape)
print ('  Means: ', out.mean(axis=(0, 2, 3)))
print ('  Stds: ', out.std(axis=(0, 2, 3)))

# Save data to matlab file
dictSaveMat={}
dictSaveMat['x']=x.astype('float')
dictSaveMat['gamma']=gamma.astype('float')
dictSaveMat['beta']=beta.astype('float')
dictSaveMat['out']=out.astype('float')
scipy.io.savemat('spatial_batchnorm_forward',dictSaveMat)