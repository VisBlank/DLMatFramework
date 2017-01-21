classdef CrossEntropy < BaseLoss
    %CROSSENTROPY Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Access = 'public')
        function [loss, gradients] = GetLossAndGradients(obj, prob, targets)
            batchScores = size(prob,1);
            % Considering that the scores are already converted to
            % probabilities.
            loss = -log(prob(find(targets)));
            loss = loss / batchScores;
            
            % Get gradient of loss w.r.t to the correct score
            gradients = prob;
            gradients(find(targets)) = gradients(find(targets)) - 1; 
            gradients = gradients / batchScores;
        end
    end
    
end

