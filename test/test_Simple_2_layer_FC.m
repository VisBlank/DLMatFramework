%% Test the creation/training of a 2 layer 2 class(not binary) classifier

clear all;
layers = LayerContainer();    
layers <= struct('name','ImageIn','type','input','rows',1,'cols',2,'depth',1, 'batchsize',1);
layers <= struct('name','FC_1','type','fc', 'num_output',100);
layers <= struct('name','Relu_1','type','relu');
layers <= struct('name','FC_2','type','fc','num_output',2);
layers <= struct('name','Softmax','type','softmax');

net = DeepLearningModel(layers, LossFactory.GetLoss('cross_entropy'));

net.Predict([1 2]);
[loss, gradients] = net.Loss([1 2], [0 1]);

solver = Solver(net, rand(1,2,1,10), 'sgd',containers.Map({'learning_rate'}, {0.1}));
solver.Train();