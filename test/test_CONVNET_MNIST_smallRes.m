%% Test the creation/training of a 2 layer 2 class(not binary) classifier
% Follow tensorflow tutorial
% https://www.tensorflow.org/tutorials/mnist/pros/
% https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
% https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
% https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/ConvolutionalNetworks.ipynb
% https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/classifiers/cnn.py
% http://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
% https://github.com/aymericdamien/TensorFlow-Examples

clear all;

% Reset random number generator state, this is needed in order to make the
% weight initialization go work
rng(0,'v5uniform');

%% Load data
load mnist_oficial;
% Crop data to make faster to train
%input_train = input_train(1:5000,:);
%input_test = input_test(1:500,:);
%output_test_labels = output_test_labels(1:500,:);
%output_train_labels = output_train_labels(1:5000,:);

data = Dataset(single(input_train), single(output_train_labels),28,28,1,1,true);
data.AddValidation(single(input_test),single(output_test_labels),28,28,1,1,true);

% Test to display batch
%batch = data.GetBatch(10);
%implay(batch.X);

%% Create network
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',28,'cols',28,'depth',1, 'batchsize',1);
layers <= struct('name','CONV1','type','conv', 'kh',5,'kw',5,'stride',1,'pad',2,'num_output', 32); 
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','MP1','type','maxpool', 'kh',2, 'kw',2, 'stride',2); 
layers <= struct('name','CONV2','type','conv', 'kh',5,'kw',5,'stride',1,'pad',2,'num_output', 64); 
layers <= struct('name','SBN_2','type','sp_batchnorm','eps',1e-5, 'momentum', 0.9);
layers <= struct('name','Relu_2','type','relu');
layers <= struct('name','CONV3','type','conv', 'kh',5,'kw',5,'stride',1,'pad',2,'num_output', 64); 
layers <= struct('name','SBN_3','type','sp_batchnorm','eps',1e-5, 'momentum', 0.9);
layers <= struct('name','Relu_3','type','relu');
layers <= struct('name','Add_1','type','add','inputs',{{'Relu_3','MP1'}});
layers <= struct('name','MP2','type','maxpool', 'kh',14, 'kw',14, 'stride',1); 
layers <= struct('name','FC','type','fc', 'num_output',1024);
layers <= struct('name','Relu_5','type','relu');
layers <= struct('name','FC_2','type','fc','num_output',data.GetNumClasses());
layers <= struct('name','Softmax','type','softmax');

% Print structure
layers.ShowStructure();

% Create DeepLearningModel instance
net = DeepLearningModel(layers, LossFactory.GetLoss('multi_class_cross_entropy'));


%% Create solver and train
solver = Solver(net, data, 'sgd',containers.Map({'learning_rate', 'L2_reg'}, {0.1, 0}));
solver.SetBatchSize(1000);
solver.SetEpochs(10);
solver.PrintEvery(10);
solver.Train();

%% Test
testBatchSize = size(input_test,1);
figure(2);
batchValidation = data.GetValidationBatch(testBatchSize);
batchImg = gather(batchValidation.X(:,:,:,1:20));
batchImg = permute(batchImg,[2,1,3,4]);
display_MNIST_Data(reshape_row_major(batchImg,[20,784]));
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
        fprintf('Predicted %d and should be %d\n',idxScoresMax(idx),idxCorrect(idx));
        % Uncomment if you want to pause on each error
        %img = batchValidation.X;
        %imshow(img(:,:,:,idx));
        %pause;
    end    
end
errorPercentage = (errorCount*100) / testBatchSize;
fprintf('Validation Accuracy is %d percent \n',round((100-errorPercentage)));
figure;
plot(solver.GetLossHistory)
