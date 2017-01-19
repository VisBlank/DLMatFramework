classdef Relu < BaseLayer
    %RELU Summary of this class goes here
    % Reference: https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    
    properties (Access = 'protected') 
        weights
        biases
        activations
        config
        previousInput
        name
    end
    
    methods (Access = 'public')
        function [obj] = Relu(name)
            obj.name = name;
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            activations = max(0,input);
            obj.activations = activations;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            gradient  = dout .* (obj.previousInput >= 0);
        end                
    end
    
end

