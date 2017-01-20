classdef DeepLearningModel < handle
    %DEEPLEARNINGMODEL Summary of this class goes here
    % References:
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/classifiers/fc_net.py
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/classifiers/cnn.py
    
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
            scores = 0;
        end
        
        function [lossVal, gradients] = Loss(obj, X, Y)
            %% Get loss and gradient of the loss w.r.t to the scores
            %% Add regularization to loss
            %% Backprop
            %% Return loss and gradients
            lossVal = [];
            gradients = [];
        end
    end
    
end

