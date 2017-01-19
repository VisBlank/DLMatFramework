classdef FullyConnected < BaseLayer
    %FULLYCONNECTED Summary of this class goes here
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
        function [obj] = FullyConnected(name)
            obj.name = name;            
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            
            % Tensor format (rows,cols,channels, batch) on matlab
            % Get batch size
            N = size(input,ndims(input));
            
            % Reshape input to have N rows and as much cols needed
            input_reshape = reshape(input,N,[]);
            activations = input_reshape*weights + bias;
            
            % Cache results for backpropagation
            obj.activations = activations;
            obj.weights = weights;
            obj.biases = bias;
            obj.previousInput = input;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            % Recover cache
            N = size(obj.previousInput,ndims(obj.previousInput));
            
            input_reshape = reshape(obj.previousInput,N,[]);
            
            gradient.bias = sum(dout);
            gradient.weight = input_reshape' * dout;
            dx = (dout * obj.weights');
            dx = resize(dx,size(obj.previousInput));
            gradient.input = dx;
        end                
    end
    
end

