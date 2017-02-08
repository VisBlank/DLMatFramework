function [ output_args ] = display_CIFAR_Data( data, batch_size )
%DISPLAY_CIFAR_DATA Summary of this function goes here
%   Detailed explanation goes here
output_args = [];

% Calculate X,Y grid size
gridX = ceil(sqrt(batch_size));
gridY = gridX;

numOver = floor(((gridX*gridY) - batch_size) / gridX);
gridY = gridY - numOver;

% The cifar consist of 10000 32x32x3 images
X = data(1:batch_size,:);
X_reshape = reshape_row_major(X,[32,32,3,batch_size]);
1+1;

%[X2,map2] = imread('forest.tif');
%subplot(1,2,1), subimage(X)
%subplot(1,2,2), subimage(X2)
close all;
for idx=1:batch_size
    subplot(gridY,gridX,idx), subimage(X_reshape(:,:,:,idx));
end

end

