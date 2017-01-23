%% Test the creation/training of a 2 layer binary classifier

% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];

% Reset random number generator state
rng(0,'v5uniform');

clear all;
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',2,'depth',1, 'batchsize',1);
layers <= struct('name','FC_1','type','fc', 'num_output',20);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc', 'num_output',1);
layers <= struct('name','SigmoidOut','type','sigmoid');

net = DeepLearningModel(layers, LossFactory.GetLoss('cross_entropy'));

net.Predict([1 2]);
[loss, gradients] = net.Loss([1 2], [0 1]);

solver = Solver(net, rand(1,2,1,10), 'sgd',containers.Map({'learning_rate'}, {0.1}));
solver.Train();