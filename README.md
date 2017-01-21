# Deep Learning Matlab Framework

## Introduction
Implementation of Deeplearning library based on my books:
* https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/
* https://leonardoaraujosantos.gitbooks.io/opencl/content/

### Example from command line
```matlab
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
