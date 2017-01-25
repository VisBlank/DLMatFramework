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
layers <= struct('name','sigmoid_1','type','sigmoid');
layers <= struct('name','FC_2','type','fc', 'num_output',1);
layers <= struct('name','SigmoidOut','type','sigmoid');

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('cross_entropy'));

% Create references for weight/bias map
weightsMap = net.getWeights();
biasMap = net.getBias();

% Fix a starting point (Initial weights) to compare different
% implementation
weightsMap('FC_1') = [0.1709   -0.0224; 0.6261    0.4194];
weightsMap('FC_2') = [-0.7704    0.5143]';
biasMap('FC_1') = [0.7202   -0.4302];
biasMap('FC_2') =  -0.0697;

% Enable gradient check (Not working yet with mini-batch)
net.EnableGradientCheck(true);

%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate'}, {0.1}));
solver.SetBatchSize(1); % or 4 for the whole dataset (Normal gradient descent)
solver.SetEpochs(1000);
solver.Train();

% Use pre-trained weights (AndrewNg based backprop)
% W1_B1 = [1.8508 -3.3049 3.5370; -1.7704 -3.2914 3.0244]
% W2_B2 = [1.5793   -3.6822    4.0742]
% FC = X * W'
% Does the bias trick
% IN_ext = [1 In1 In2]
% W_ext = [b W]
%weightsMap('FC_1') = [-3.3049   -3.2914; 3.5370    3.0244];
%weightsMap('FC_2') = [-3.6822    4.0742]';

%biasMap('FC_1') = [1.8508   -1.7704];
%biasMap('FC_2') =  1.5793;

%% Test
[scores] = net.Predict([0 0]);
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[scores] = net.Predict([0 1]);
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[scores] = net.Predict([1 0]);
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[scores] = net.Predict([1 1]);
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

% Plot Prediction surface
testInpx1 = [-0.5:0.1:1.5];
testInpx2 = [-0.5:0.1:1.5];
[X1, X2] = meshgrid(testInpx1, testInpx2);
testOutRows = size(X1, 1);
testOutCols = size(X1, 2);
testOut = zeros(testOutRows, testOutCols);
for row = [1:testOutRows]
    for col = [1:testOutCols]
        test = [X1(row, col), X2(row, col)];
        %% Forward pass     
        [A3] = net.Predict(test);
        testOut(row, col) = A3;
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');

figure(1);
plot(solver.GetLossHistory);
title('Loss');