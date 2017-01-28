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
layers <= struct('name','FC_1','type','fc', 'num_output',200);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc', 'num_output',200);
layers <= struct('name','Relu_2','type','relu');
layers <= struct('name','FC_3','type','fc', 'num_output',200);
layers <= struct('name','Relu_3','type','relu');
layers <= struct('name','FC_4','type','fc', 'num_output',200);
layers <= struct('name','Relu_4','type','relu');
layers <= struct('name','FC_20','type','fc','num_output',data.GetNumClasses());
layers <= struct('name','Softmax','type','softmax');

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('multi_class_cross_entropy'));

% Create references for weight/bias map
layerCont = net.getLayers();
weightsMap = net.getWeights();
biasMap = net.getBias();


%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate', 'L2_reg', 'lr_decay'}, {0.5, 0, 0.9999})); % Without L2 Reg
solver.SetBatchSize(300);
solver.SetEpochs(5000);
solver.Train();

figure(1);
plot(solver.GetLossHistory);
title('Loss training');

%% Display prediction curves with original data
% Some references between numpy and matlab
% http://mathesaurus.sourceforge.net/matlab-numpy.html
% http://mathesaurus.sourceforge.net/matlab-python-xref.pdf
% http://cheatsheets.quantecon.org/
h = 0.02;
% Get min and max values from coordinates of X
[x_minMax] = minmax(X(:,1)');[y_minMax] = minmax(X(:,2)');
x_min = x_minMax(1) - 1;x_max = x_minMax(2) + 1;
y_min = y_minMax(1) - 1;y_max = y_minMax(2) + 1;

[xx, yy] = meshgrid([x_min:h:x_max],[y_min:h:y_max]);

% Flat xx and yy and create X_test
X_test = [xx(:), yy(:)];

Z = net.Predict(X_test);
% Get before softmax
%Z = layerCont('FC_2').getActivations;
[~,Z] = max(Z,[],2);
Z = reshape(Z,size(xx));

% Plot 
figure(2);
alpha(.5);
contourf(xx, yy,Z);
axis([-1.1 1.1 y_min y_max]);
hold on;
gscatter(X(:,1), X(:,2), Y);
title('Prediction Surface');
hold off;

