%% Test the creation/training of a 2 layer 2 class(not binary) classifier

clear all;
%% Load data
load mnist_oficial;
data = Dataset(input_train, output_train,1,784,1,1);

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',784,'depth',1, 'batchsize',10);
layers <= struct('name','FC_1','type','fc', 'num_output',500);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc','num_output',10);
layers <= struct('name','Softmax','type','softmax');

net = DeepLearningModel(layers, LossFactory.GetLoss('cross_entropy'));

net.Predict([1 2]);
[loss, gradients] = net.Loss([1 2], [0 1]);

%% Create solver
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate'}, {0.1}));
solver.Train();