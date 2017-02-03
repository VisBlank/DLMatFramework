

import numpy as np
import scipy.io
from cs231n.im2col import *
from time import time


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))



# Load from matfile the parameters
dictMat = scipy.io.loadmat('../../../test/layers/someBatches.mat')
# On python is (N,C,H,W)
# On matlab is (H,W,C,N)
batch_1 = dictMat['batch_4_4_3_2']
batch_2 = dictMat['batch_5_5_3_2']

pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# As those tensors come from matlab we need to permute their dimensions
batch_1 = batch_1.transpose(3, 2, 0, 1)
batch_2 = batch_2.transpose(3, 2, 0, 1)

# Do the im2col with the smaller batch (Stride 2, K=2)
batch_1_col = im2col_slow(batch_1, pool_param['pool_height'], pool_param['pool_width'], 0, pool_param['stride'])

# Do the im2col with the bigger batch (Stride 1, K=3, Pad=1)
batch_2_col = im2col_slow(batch_2, 3, 3, 1, 1)

# We want to use a conv layer with kernel(3,3) S:1, Pad:1, input volume:3 output Volume(F):3
# Create unit kernel (One filter per channel, same filter to each batch
#kernel_ch1 = np.matrix('0 0 0; 0 0.1 0; 0 0 0')
#kernel_ch2 = np.matrix('0 0 0; 0 0.1 0; 0 0 0')
#kernel_ch3 = np.matrix('0 0 0; 0 0.1 0; 0 0 0')
#kernel_batch_1 = np.concatenate((kernel_ch1.ravel(),kernel_ch2.ravel(),kernel_ch3.ravel()), axis=1)
#kernel_batch_2 = kernel_batch_1
#kernel_batch_3 = kernel_batch_1
#kernel = np.concatenate((kernel_batch_1,kernel_batch_2,kernel_batch_3), axis=0)

# Apply kernel to resul of im2col
#conv_result = kernel.dot(batch_2_col)
#conv_result = np.asarray(conv_result)

# Inverse with col2im (N,C,H,W,k_h,k_w, padding, stride)
#batch_1_rev = conv_result.reshape(3, 5, 5, 2)
#batch_1_rev = batch_1_rev.transpose(3, 0, 1, 2)

#batch_1_rev = col2im_slow(conv_result, 2,3,5,5,3,3,0,1)

1+1