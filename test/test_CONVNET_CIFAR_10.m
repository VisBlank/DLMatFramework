clear all;

% Reset random number generator state, this is needed in order to make the
% weight initialization go work
rng(0,'v5uniform');

%% Load data
load complete_cifar_10;
