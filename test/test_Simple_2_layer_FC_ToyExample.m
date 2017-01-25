%% Test the creation/training of a 2 layer 2 class(not binary) classifier
clear all;
close all;

% Reset random number generator state, this is needed in order to make the
% weight initialization go work
rng(0,'v5uniform');

%% Load data
load ToyExample;
data = Dataset(X, Y',1,2,1,1,true);
%data.pushToGPU();

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',2,'depth',1, 'batchsize',1);
layers <= struct('name','FC_1','type','fc', 'num_output',100);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc','num_output',data.GetNumClasses());
layers <= struct('name','Softmax','type','softmax');

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('multi_class_cross_entropy'));


%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate'}, {0.2}));
solver.SetBatchSize(10);
solver.SetEpochs(10000);
solver.Train();

plot(solver.GetLossHistory)
