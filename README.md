# Deep Learning Matlab Framework

## Introduction
Implementation of Deeplearning library based on my books:
* https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/
* https://leonardoaraujosantos.gitbooks.io/opencl/content/

### Example from command line
```matlab
%% Test the creation/training of a 2 layer 2 class(not binary) classifier
clear all;
% Reset random number generator state, this is needed in order to make the
% weight initialization go work
rng(0,'v5uniform');

%% Load data (MNIST)
load mnist_oficial;
data = Dataset(input_train, output_train_labels,1,784,1,1);
data.AddValidation(input_test,output_test_labels,1,784,1,1);

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',784,'depth',1, 'batchsize',1);
layers <= struct('name','FC_1','type','fc', 'num_output',50);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc','num_output',data.GetNumClasses());
layers <= struct('name','Softmax','type','softmax');

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('cross_entropy'));

%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate'}, {0.01}));
solver.SetBatchSize(64);
solver.SetEpochs(100);
solver.Train();

%% Test
figure(2);
batchValidation = data.GetValidationBatch(10);
display_MNIST_Data(reshape_row_major(batchValidation.X,[10,784]));
title('Images on validation');
errorCount = 0;

% Predict the batch
scores = net.Predict(batchValidation.X);
[~, idxScoresMax] = max(scores,[],2);
[~, idxCorrect] = max(batchValidation.Y,[],2);
% Subtract one (First class )
idxScoresMax = idxScoresMax - 1;
idxCorrect = idxCorrect - 1;

% Compare scores with target
for idx=1:10    
    if idxScoresMax(idx) ~= idxCorrect(idx)
        errorCount = errorCount + 1;
        fprintf('Predicted %d and should be %d\n',idxScoresMax(idx),idxCorrect(idx));
    end    
end
errorPercentage = (errorCount*100) / 10;
fprintf('Accuracy is %d percent \n',round((100-errorPercentage)));

plot(solver.GetLossHistory)
```

### Objectives
The idea is to create a library as readable as possible while maintaning usability by giving the following features
* C/C++ code generation support
* Support for low-end platforms like raspbery PI
* Allows graphical representation of models on simulink
* GPU implementation on both CUDA and OpenCL
* Usage of matlab distributed features for scaling and performance

### Tutorials or references
All documentation will be available through my books or youtube channel. I will add tutorials as needed.
