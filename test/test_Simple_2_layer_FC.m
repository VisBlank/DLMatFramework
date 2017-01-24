%% Test the creation/training of a 2 layer binary classifier
clc; clear all;

% Reset random number generator state
rng(0,'v5uniform');

%% Load data
% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];
Xt = [0 0; 0 1; 1 0; 1 1];
Yt = [ 0; 1; 1; 0];

data = Dataset(X, Y,1,2,1,1,false);
data.AddValidation(Xt,Yt,1,2,1,1,false);

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',2,'depth',1, 'batchsize',1);
layers <= struct('name','FC_1','type','fc', 'num_output',2);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc', 'num_output',1);
layers <= struct('name','SigmoidOut','type','sigmoid');

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('cross_entropy'));

% Enable gradient check
net.EnableGradientCheck(true);

%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate'}, {0.01}));
solver.SetBatchSize(1); % or 4 for the whole dataset (Normal gradient descent)
solver.SetEpochs(10);
solver.Train();

%% Test
batchValidation = data.GetValidationBatch(4);
[scores] = net.Predict(batchValidation.X);

% [scores] = net.Predict(Xt(1,:));
% fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
% [scores] = net.Predict(Xt(2,:));
% fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
% [scores] = net.Predict(Xt(3,:));
% fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
% [scores] = net.Predict(Xt(4,:));
% fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

plot(solver.GetLossHistory);