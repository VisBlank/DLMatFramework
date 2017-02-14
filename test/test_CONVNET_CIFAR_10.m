clear all;

% Reset random number generator state, this is needed in order to make the
% weight initialization go work
rng(0,'v5uniform');

%% Load data
load complete_cifar_10;
load cifar_10_test;

data = Dataset(single(data), single(labels),32,32,3,1,true);
data.AddValidation(single(data_test),single(labels_test),32,32,3,1,true);

% Calculate the mean image/pixel on the whole set
imgMean = getMeanImageOnBatch(data.GetBatch(-1).X);
pixelMean = getMeanPixelOnBatch(data.GetBatch(-1).X);
% For cifar-10: {125.33941,122.96881,113.90228}

% Enable Augmentation during training
data.enableAugmentation(true);
% Enable Mean pixel normalization
data.enableMeanPixelNormalization(true,pixelMean);

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',32,'cols',32,'depth',3, 'batchsize',1);
layers <= struct('name','CONV1','type','conv', 'kh',5,'kw',5,'stride',1,'pad',2,'num_output', 32); 
layers <= struct('name','SBN_1','type','sp_batchnorm','eps',1e-5, 'momentum', 0.9);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','MP1','type','maxpool', 'kh',2, 'kw',2, 'stride',2); 
layers <= struct('name','CONV2','type','conv', 'kh',5,'kw',5,'stride',1,'pad',2,'num_output', 64); 
layers <= struct('name','SBN_2','type','sp_batchnorm','eps',1e-5, 'momentum', 0.9);
layers <= struct('name','Relu_2','type','relu');
layers <= struct('name','MP2','type','maxpool', 'kh',2, 'kw',2, 'stride',2); 
layers <= struct('name','FC_3','type','fc', 'num_output',1024);
layers <= struct('name','BN_3','type','batchnorm','eps',1e-5, 'momentum', 0.9);
layers <= struct('name','Relu_3','type','relu');
layers <= struct('name','DRP_1','type','dropout','prob',0.5);
layers <= struct('name','FC_4','type','fc','num_output',data.GetNumClasses());
layers <= struct('name','Softmax','type','softmax');

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('multi_class_cross_entropy'));


%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate', 'L2_reg'}, {0.1, 0}));
solver.SetBatchSize(5000);
solver.SetEpochs(250);
solver.PrintEvery(10);
solver.Train();

%% Test
batchValidation = data.GetValidationBatch(-1);
testBatchSize = size(batchValidation.Y,1);

% Predict the batch
scores = net.Predict(batchValidation.X);
[~, idxScoresMax] = max(scores,[],2);
[~, idxCorrect] = max(batchValidation.Y,[],2);
% Subtract one (First class )
idxScoresMax = idxScoresMax - 1;
idxCorrect = idxCorrect - 1;
errorCount = 0;

% Compare scores with target
for idx=1:testBatchSize    
    if idxScoresMax(idx) ~= idxCorrect(idx)
        errorCount = errorCount + 1;
        %fprintf('Predicted %d and should be %d\n',idxScoresMax(idx),idxCorrect(idx));
        % Uncomment if you want to pause on each error
        %img = batchValidation.X;
        %imshow(img(:,:,:,idx));
        %pause;
    end    
end
errorPercentage = (errorCount*100) / testBatchSize;
fprintf('Validation Accuracy is %d percent \n',round((100-errorPercentage)));

%% Plot loss history
figure;
plot(solver.GetLossHistory)