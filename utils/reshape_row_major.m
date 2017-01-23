function [ reshaped_matrix ] = reshape_row_major( matrix_in, shape )
%RESHAPE_ROW_MAJOR Do the reshape with row_major scan
% This is needed because other numerical libraries like numpy used
% different order (row_major) compared to matlab(col_major)
% Example:
% a = [1:1:15]
% reshape(a,[3,5])
%  1     4     7    10    13
%  2     5     8    11    14
%  3     6     9    12    15
%
% reshape_row_major(a,[3,5])
%  1     2     3     4     5
%  6     7     8     9     10
%  11    12    13    14    15

% Transpose the input
% Only rely on permute which can handle bigger dimensions compared to (')
matrix_in =  permute(matrix_in,[2 1 3 4]);

% Flip shape, but supporting tensors (Just flip the first 2 dimensions)
flipped_shape = shape;
flipped_shape(1) = shape(2); flipped_shape(2) = shape(1);
res_trans = reshape(matrix_in,flipped_shape);

% Transpose the tensor (Vectors, until 4D)
% Just permute the first 2 dimensions
reshaped_matrix =  permute(res_trans,[2 1 3 4]);

end