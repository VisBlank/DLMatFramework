classdef DeepLearningModel < handle
    %DEEPLEARNINGMODEL Summary of this class goes here
    % References:
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/classifiers/fc_net.py
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/classifiers/cnn.py
    % Ex:    
    % layers = LayerContainer();    
    % layers <= struct('name','FC_1','type','fc');
    % layers <= struct('name','Relu_1','type','relu');
    % layers <= struct('name','FC_2','type','fc');
    % layers <= struct('name','Softmax','type','softmax');
    % layers.getNumLayers();
    % net = DeepLearningModel(layers)
    
    properties (Access = 'protected')
        layersContainer
        lossFunction
    end
    
    methods (Access = 'public')
        function obj = DeepLearningModel(layerCont, lossType)
            obj.layersContainer = layerCont;
            obj.lossFunction = LossFactory.GetLoss(lossType);
            %% Initialize weights and biases
        end
        
        function [scores] = Predict(obj, X)
            % Get input signals coordinates
            [rows,cols,depth,batch] = size(X);
            
            % Iterate forward on the graph
            currInput = X;
            for idxLayer=1:obj.layersContainer.getNumLayers()
                currLayer = obj.layersContainer.getLayerFromIndex(idxLayer);
                currInput = currLayer.ForwardPropagation(currInput,weight,bias);                
            end
            scores = currInput;
        end
        
        function [lossVal, gradients] = Loss(obj, X, Y)
            %% Get loss and gradient of the loss w.r.t to the scores
            %% Add regularization to loss
            %% Backprop
            %% Return loss and gradients
            lossVal = [];
            gradients = [];
        end
        
        function ShowStructure(obj)
           obj.layersContainer.ShowStructure(); 
        end
    end
    
end

