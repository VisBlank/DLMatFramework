function [ img_grad ] = im2col_back_custom( dout, img_grad_before_pad_H, img_grad_before_pad_W, S, kH, kW, C )

% Use this stub function to change implementations, for example change
% between vanilla matlab only code to mex Using OpenCl/Cuda or OpenMP
%img_grad = im2col_back_ref( dout, img_grad_before_pad_H, img_grad_before_pad_W, S, kH, kW, C );

% Uncomment for using mex version
[img_grad,execTime,trfTime] = mex_im2col_back( dout, img_grad_before_pad_H, img_grad_before_pad_W, S, kH, kW, C );
end