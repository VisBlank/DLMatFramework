clear all;

% Reset random number generator state, this is needed in order to make the
% weight initialization go work
rng(0,'v5uniform');

%% Load data
load complete_cifar_10;
load cifar_10_test;

data = Dataset(single(data), single(labels),32,32,3,1,true);
data.AddValidation(single(data_test),single(labels_test),32,32,3,1,true);

% Display a batch sample
%batch = data.GetBatch(10);

% Calculate the mean image/pixel on the whole set
imgMean = getMeanImageOnBatch(data.GetBatch(-1).X);
pixelMean = getMeanPixelOnBatch(data.GetBatch(-1).X);
% For cifar-10: {125.33941,122.96881,113.90228}

