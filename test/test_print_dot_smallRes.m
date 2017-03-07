%% Test DOT graph creation

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

