clc; clear all;
load im2col_with_batch_cs231n

% Fix the tensor dimensions between python and matlab
x = permute(x,[3,4,2,1]);
%x = x(:,:,:,1);

% Pass values to double
P = double(pad);
S = double(stride);
HH = double(filter_height);
WW = double(filter_width);

% The input x will be convolved with 3x3 S:1 P:1, the tensor has 3 channels
% and 4 images on the batch

% Get tensor dimensions
[rows, cols, C, N] = size(x);

% Calculate convolved result size.
convOut_H = ((rows + 2*P - HH) / S)+1;
convOut_W = ((cols + 2*P - WW) / S)+1;

%im_col_complete = zeros((C*HH*WW), (convOut_H * convOut_W * N));
im_col_complete = [];
% Check results with im2col no batch
for idxBatch = 1:N
    im = x(:,:,:,idxBatch);
    im_col = im2col_ref(im,HH,WW,S,P,1);
    im_col_complete = [im_col_complete im_col];    
end

diff = sum(abs(x_cols(:) - im_col_complete(:)));
