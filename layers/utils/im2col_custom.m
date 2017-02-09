function [ img_matrix ] = im2col_custom( inputImg, k_height, k_width, S , P)

% Use this stub function to change implementations, for example change
% between vanilla matlab only code to mex Using OpenCl/Cuda or OpenMP
%img_matrix = im2col_ref( inputImg, k_height, k_width, S , P, 1 );

% Uncomment for using mex version
[img_matrix,~,~] = mex_im2col(inputImg, k_height, k_width, S , P); 
end