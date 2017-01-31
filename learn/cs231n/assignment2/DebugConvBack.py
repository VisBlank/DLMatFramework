

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver


def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul,h_prime,w_prime,C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
    """
    Args:
      dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
      h_prime,w_prime: height and width for the feature map
      strid: stride
      hh,ww,c: size of the filters
    Returns:
      dx: Gradients for x, (C,H,W)
    """
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c,H,W])
    for i in range(h_prime*w_prime):
        row = dim_col[i,:]
        h_start = (i / w_prime) * stride
        w_start = (i % w_prime) * stride
        patch = np.reshape(row, (c, hh, ww))
        dx[:,h_start:h_start+hh,w_start:w_start+ww] += patch
    return dx

def conv_forward_naive_im2col(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad_num = conv_param['pad']
  stride = conv_param['stride']
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_prime = (H+2*pad_num-HH) // stride + 1
  W_prime = (W+2*pad_num-WW) // stride + 1
  out = np.zeros([N,F,H_prime,W_prime])
  #im2col
  for im_num in range(N):
      im = x[im_num,:,:,:]
      im_pad = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
      im_col = im2col(im_pad,HH,WW,stride)
      filter_col = np.reshape(w,(F,-1))
      mul = im_col.dot(filter_col.T) + b
      out[im_num,:,:,:] = col2im(mul,H_prime,W_prime,1)
  cache = (x, w, b, conv_param)
  return out, cache

def conv_backward_naive_im2col(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  x, w, b, conv_param = cache
  pad_num = conv_param['pad']
  stride = conv_param['stride']
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_prime = (H+2*pad_num-HH) // stride + 1
  W_prime = (W+2*pad_num-WW) // stride + 1

  # Initialize variables
  dw = np.zeros(w.shape)
  dx = np.zeros(x.shape)
  db = np.zeros(b.shape)

  # Get bias gradient
  # Bias gradient (Sum on dout dimensions (batch, rows, cols)
  db = np.sum(dout, axis=(0, 2, 3))

  for i in range(N):
      im = x[i,:,:,:]
      im_pad = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
      im_col = im2col(im_pad,HH,WW,stride)
      filter_col = np.reshape(w,(F,-1)).T

      dout_i = dout[i,:,:,:]
      dbias_sum = np.reshape(dout_i,(F,-1))
      dbias_sum = dbias_sum.T

      #bias_sum = mul + b
      #db += np.sum(dbias_sum,axis=0)
      dmul = dbias_sum

      #mul = im_col * filter_col
      dfilter_col = (im_col.T).dot(dmul)
      dim_col = dmul.dot(filter_col.T)

      # im2col Backward propagation
      dx_padded = col2im_back(dim_col,H_prime,W_prime,stride,HH,WW,C)

      # Take out padding
      dx[i,:,:,:] = dx_padded[:,pad_num:H+pad_num,pad_num:W+pad_num]

      dw += np.reshape(dfilter_col.T,(F,C,HH,WW))
  return dx, dw, db

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# # Convolution: Naive backward pass
# Implement the backward pass for the convolution operation in the function `conv_backward_naive` in the file `cs231n/layers.py`. Again, you don't need to worry too much about computational efficiency.
# 
# When you are done, run the following to check your backward pass with a numeric gradient check.

# In[14]:

# Load from matfile the parameters
dictMat = scipy.io.loadmat('../../../test/layers/conv_backward_cs231n.mat')
x = dictMat['x']
w = dictMat['w']
b = dictMat['b']
dout = dictMat['dout']

conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

# Test the conv with im2col version of forward/backward propagation
out, cache = conv_forward_im2col(x, w, b, conv_param)
dx, dw, db = conv_backward_im2col(dout, cache)

# Your errors should be around 1e-9'
print ('Testing conv_backward_naive function')
print ('dx error: ', rel_error(dx, dx_num))
print ('dw error: ', rel_error(dw, dw_num))
print ('db error: ', rel_error(db, db_num))

out, cache = conv_forward_naive_im2col(x, w, b, conv_param)
dx, dw, db = conv_backward_naive_im2col(dout, cache)
print ('Testing conv_backward_naive_im2col function')
print ('dx error: ', rel_error(dx, dx_num))
print ('dw error: ', rel_error(dw, dw_num))
print ('db error: ', rel_error(db, db_num))



# Some tests with im2col (Preparing the image x to be convolved with a 3,3 kernel with stride:1 pad:1
H = 5
W = 5
filter_height = 3
filter_width = 3
pad = 1
stride = 1

# Calculate spatial output sizes
out_height = (H + 2 * pad - filter_height) / stride + 1
out_width = (W + 2 * pad - filter_width) / stride + 1

print("original x: ", x.shape)
x_cols = im2col_cython(x, filter_height, filter_width, pad, stride)
print('im2col result: ', x_cols.shape)
print('Conv out height: %d width: %d' % (out_height, out_width))

# Save vectors to matlab for tests
import scipy.io

dictSaveMat={}
dictSaveMat['x']=x.astype('float')
dictSaveMat['x_cols']=x_cols.astype('float')
dictSaveMat['pad']=pad
dictSaveMat['stride']=stride
dictSaveMat['filter_height']=filter_height
dictSaveMat['filter_width']=filter_width
scipy.io.savemat('im2col_with_batch_cs231n',dictSaveMat)

dictMat = scipy.io.loadmat('SimpleInput.mat')
x_simple = dictMat['x_simple']
x_simple = x_simple.transpose(3,2,0,1)
x_simple = x_simple.astype('float')
x_simple = x_simple.reshape(2, 3, 4, 4)
x_cols_simple = im2col_cython(x_simple, 2, 2, 0, 1)

# This im2col does not accept multiple batches
x_cols_simple_2 = im2col(x_simple[0,:,:,:], 2, 2, 1)
# Transpose, and it will be the same as the docs.
x_cols_simple_2 = x_cols_simple_2.T
1+1