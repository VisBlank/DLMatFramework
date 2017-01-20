classdef MeanSquaredError < BaseLoss
    %MEANSQUAREDERROR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Access = 'public')
        function [loss, gradients] = GetLossAndGradients(obj, scores, targets)
            loss = [];
            gradients = [];
        end
    end
    
end

