

import numpy as np
import scipy.io
from cs231n.layers import *
from cs231n.fast_layers import *
from time import time


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# # Convolution: Naive backward pass
# Implement the backward pass for the convolution operation in the function `conv_backward_naive` in the file `cs231n/layers.py`. Again, you don't need to worry too much about computational efficiency.
# 
# When you are done, run the following to check your backward pass with a numeric gradient check.

# In[14]:

# Load from matfile the parameters
dictMat = scipy.io.loadmat('../../../test/layers/maxpool_backward_cs231n.mat')
x = dictMat['x']
dx_num = dictMat['dx_num']
dout = dictMat['dout']

pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# Test max_pool_forward/backward naive
t0 = time()
out, cache = max_pool_forward_naive(x, pool_param)
dx = max_pool_backward_naive(dout, cache)
t1 = time()
timeNaive = t1-t0
print ('Forward/Backward maxpool naive: %fs' % (timeNaive))
print ('Testing max_pool naive functions')
print ('dx error: ', rel_error(dx, dx_num))

print ('Shape of x', x.shape)
print ('Shape of dx', dx.shape)
print ('Shape of out', out.shape)
print ('Shape of dout', dout.shape)

# Test max_pool_forward/backward fast
t0 = time()
out, cache = max_pool_forward_fast(x, pool_param)
dx = max_pool_backward_fast(dout, cache)
t1 = time()
timeFast = t1-t0
print ('Forward/Backward maxpool fast: %fs' % (timeFast))
print ('Speedup: %fx' % (timeNaive / timeFast))

# Your errors should be around 1e-9'
print ('Testing max_pool fast functions')
print ('dx error: ', rel_error(dx, dx_num))







