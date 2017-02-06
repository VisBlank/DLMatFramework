

import numpy as np
import scipy.io
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


############################### FORWARD TRAIN ############################################
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

############################### BACKWARD ############################################
# Check backward pass
N, C, H, W = 2, 3, 4, 5
#x = 5 * np.random.randn(N, C, H, W) + 12
#gamma = np.random.randn(C)
#beta = np.random.randn(C)
#dout = np.random.randn(N, C, H, W)

# Load data from matlab file to help debugging
dictMat = scipy.io.loadmat('../../../test/layers/spatial_batchnorm_backward.mat')
x = dictMat['x']
gamma = dictMat['gamma']
beta = dictMat['beta']
dout = dictMat['dout']

bn_param = {'mode': 'train'}
fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma, dout)
db_num = eval_numerical_gradient_array(fb, beta, dout)

_, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
print ('dx error: ', rel_error(dx_num, dx))
print ('dgamma error: ', rel_error(da_num, dgamma))
print ('dbeta error: ', rel_error(db_num, dbeta))

# Save data to matlab file
dictSaveMat={}
dictSaveMat['x']=x.astype('float')
dictSaveMat['gamma']=gamma.astype('float')
dictSaveMat['beta']=beta.astype('float')
dictSaveMat['dx_num']=dx_num.astype('float')
dictSaveMat['dgamma_num']=da_num.astype('float')
dictSaveMat['dbeta_num']=db_num.astype('float')
dictSaveMat['dout']=dout.astype('float')
scipy.io.savemat('spatial_batchnorm_backward',dictSaveMat)