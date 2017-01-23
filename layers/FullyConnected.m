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
        index
        activationShape
        inputLayer
        numOutput
    end
    
    methods (Access = 'public')
        function [obj] = FullyConnected(name, numOutput, index, inLayer)
            obj.name = name;   
            obj.index = index;
            obj.numOutput = numOutput;
            obj.inputLayer = inLayer;
            obj.activationShape = [1 numOutput];
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            
            % Tensor format (rows,cols,channels, batch) on matlab
            % Get batch size
            %[rows,cols,depth,N] = size(input);      
            lenSizeActivations = length(size(input));
            if (lenSizeActivations < 3)
                N = size(input,1);
            else
                N = size(input,ndims(input)); 
            end
            
            % Reshape input to have N rows and as much cols needed
            input_reshape = reshape(input,N,[]);
            activations = input_reshape*weights + repmat(bias,N,1);
            
            % Cache results for backpropagation
            obj.activations = activations;
            obj.weights = weights;
            obj.biases = bias;
            obj.previousInput = input;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;
            % Recover cache                        
            lenSizeActivations = length(size(obj.previousInput));
            if (lenSizeActivations < 3)
                N = size(obj.previousInput,1);
            else
                N = size(obj.previousInput,ndims(obj.previousInput)); 
            end
            
            input_reshape = reshape(obj.previousInput,N,[]);
            
            gradient.bias = sum(dout);
            gradient.weight = input_reshape' * dout;
            dx = (dout * obj.weights');
            dx = reshape(dx,size(obj.previousInput));
            gradient.input = dx;
        end    
        
        function [numOut] = getNumOutput(obj)
           numOut = obj.numOutput; 
        end                
    end
    
end

