%% Test the creation/training of a 2 layer 2 class(not binary) classifier

clear all;

% Reset random number generator state, this is needed in order to make the
% weight initialization go work
rng(0,'v5uniform');

%% Load data
load mnist_oficial;
data = Dataset(input_train, output_train_labels,1,784,1,1,true);
data.AddValidation(input_test,output_test_labels,1,784,1,1,true);
data.pushToGPU();

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',784,'depth',1, 'batchsize',1);
layers <= struct('name','FC_1','type','fc', 'num_output',50);
layers <= struct('name','BN_1','type','batchnorm','eps',1e-5, 'momentum', 0.9);
layers <= struct('name','Relu_1','type','relu');
%layers <= struct('name','FC_2','type','fc', 'num_output',10);
%layers <= struct('name','BN_2','type','batchnorm','eps',1e-5, 'momentum', 0.9);
%layers <= struct('name','Relu_2','type','relu');
%layers <= struct('name','FC_3','type','fc', 'num_output',200);
%layers <= struct('name','BN_3','type','batchnorm','eps',1e-5, 'momentum', 0.9);
%layers <= struct('name','Relu_3','type','relu');
%layers <= struct('name','DRP_1','type','dropout','prob',0.5);
layers <= struct('name','FC_4','type','fc','num_output',data.GetNumClasses());
layers <= struct('name','Softmax','type','softmax');

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('multi_class_cross_entropy'));


%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate', 'L2_reg'}, {0.01, 0}));
solver.SetBatchSize(200);
solver.SetEpochs(60);
solver.Train();

%% Test
testBatchSize = size(input_test,1);
figure(2);
batchValidation = data.GetValidationBatch(testBatchSize);
%display_MNIST_Data(reshape_row_major(batchValidation.X,[testBatchSize,784]));
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
for idx=1:testBatchSize    
    if idxScoresMax(idx) ~= idxCorrect(idx)
        errorCount = errorCount + 1;
        %fprintf('Predicted %d and should be %d\n',idxScoresMax(idx),idxCorrect(idx));
    end    
end
errorPercentage = (errorCount*100) / testBatchSize;
fprintf('Validation Accuracy is %d percent \n',round((100-errorPercentage)));
figure;
plot(solver.GetLossHistory)
