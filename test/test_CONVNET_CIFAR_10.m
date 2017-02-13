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
batch = data.GetBatch(10);

