classdef Softmax < BaseLayer
    %RELU Summary of this class goes here
    % Reference: https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    
    properties (Access = 'protected')
        weights
        biases
        activations
        config
        previousInput
        name
        index
    end
    
    methods (Access = 'public')
        function [obj] = Softmax(name, index)
            obj.name = name;
            obj.index = index;
        end
        
        function [activations] = ForwardPropagation(obj, scores, weights, bias)
            % Fix numerical error
            scoresFix = scores - repmat(max(scores,[],2),1,size(scores,2));
            
            % Get the sum of all scores
            sumProb = sum(exp(scoresFix),2);
            
            % Repeat this value for every column of scores
            sumProb = repmat(sumProb,1,size(scores,2));
            
            % Calculate probabilities
            activations = exp(scoresFix) ./ sumProb;
            
        end
        
        % The softmax activation has only one input (scores) so we just
        % need to find it's derivative to a specific score
        % http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        function [gradient] = BackwardPropagation(obj, dout)
            % TODO
            gradient = [];
        end
    end
    
end

