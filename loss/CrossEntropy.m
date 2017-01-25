classdef CrossEntropy < BaseLoss
    % Cross-Entropy loss
    % References:
    % https://www.ics.uci.edu/~pjsadows/notes.pdf
    
    properties
    end
    
    methods (Access = 'public')
        function [loss, gradients] = GetLossAndGradients(obj, prob, targets)
            N = size(prob,1);
            h = prob;
            loss = sum(sum((-targets).*log(h) - (1-targets).*log(1-h), 2))/N; 
            
            % dw is the derivative of the loss function over the scores
            gradients = prob - targets;
        end
    end    
end

