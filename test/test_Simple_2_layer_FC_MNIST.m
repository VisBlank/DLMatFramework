%% Test the creation/training of a 2 layer 2 class(not binary) classifier

clear all;
% Reset random number generator state
rng(0,'v5uniform');
%% Load data
load mnist_oficial;
data = Dataset(input_train, output_train_labels,1,784,1,1);
data.AddValidation(input_test,output_test_labels,1,784,1,1);

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',784,'depth',1, 'batchsize',1);
layers <= struct('name','FC_1','type','fc', 'num_output',500);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc','num_output',data.GetNumClasses());
layers <= struct('name','Softmax','type','softmax');

% On this structure the weights should be:
% W_FC1 = [784x500]
% W_FC2 = [??x10]
net = DeepLearningModel(layers, LossFactory.GetLoss('cross_entropy'));


%% Create solver
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate'}, {0.1}));
solver.Train();