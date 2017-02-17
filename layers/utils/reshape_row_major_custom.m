function [ reshaped_matrix ] = reshape_row_major_custom( matrix_in, shape )
%RESHAPE_ROW_MAJOR_CUSTOM Summary of this function goes here
%   Detailed explanation goes here
[ reshaped_matrix ] = reshape_row_major( matrix_in, shape );

% TODO: Investigate crash on MNIST example
%[reshaped_matrix,execTime,trfTime] = mex_reshape_row_major(matrix_in,shape);

end

