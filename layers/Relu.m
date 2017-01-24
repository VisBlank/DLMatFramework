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
        index
        activationShape
        inputLayer
    end
    
    methods (Access = 'public')
        function [obj] = Relu(name, index, inLayer)
            obj.name = name;
            obj.index = index;
            obj.inputLayer = inLayer;
            % Relu does not change the shape of it's output
            obj.activationShape = obj.inputLayer.getActivationShape();
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            activations = max(0,input);
            obj.activations = activations;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;
            dx = dout .* (obj.previousInput >= 0);
            gradient.input = dx;
        end
        
        function gradient = EvalBackpropNumerically(obj, dout)
            % Fully connected layers has 3 inputs so we have 3 gradients
            relu_x = @(x) obj.ForwardPropagation(x,obj.weights, obj.biases);            
            
            % Evaluate
            gradient.input = GradientCheck.Eval(relu_x,obj.previousInput) .* dout;            
        end
    end
    
end

