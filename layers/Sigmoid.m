classdef Sigmoid < BaseLayer
    %Sigmoid Summary of this class goes here
    % Reference: https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    
    properties (Access = 'protected') 
        weights
        biases
        activations
        config
        previousInput
        name
        index
        activationShape
        inputLayer
    end
    
    methods (Access = 'public')
        function [obj] = Sigmoid(name, index, inLayer)
            obj.name = name;
            obj.index = index;
            obj.inputLayer = inLayer;
            % Relu does not change the shape of it's output
            obj.activationShape = obj.inputLayer.getActivationShape();
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            activations = (1./(1+exp(-input)));
            obj.activations = activations;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;
            t = (1./(1+exp(-obj.previousInput)));
            d_sigm  = (t .* (1 - t));            
            dx = dout .* d_sigm;
            gradient.input = dx;
        end                
    end
    
end

