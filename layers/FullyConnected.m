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
            
            if obj.doGradientCheck
                evalGrad = obj.EvalBackpropNumerically(dout);
                diff_Input = sum(abs(evalGrad.input(:) - gradient.input(:)));
                diff_Weights = sum(abs(evalGrad.weight(:) - gradient.weight(:)));
                diff_Bias = sum(abs(evalGrad.bias(:) - gradient.bias(:)));
                diff_vec = [diff_Input diff_Weights ]; % diff_Bias
                diff = sum(diff_vec);
                if diff > 0.0001
                    msgError = sprintf('%s gradient failed!\n',obj.name);
                    error(msgError);
                else
                    %fprintf('%s gradient passed!\n',obj.name);
                end
            end
        end
        
        function [numOut] = getNumOutput(obj)
            numOut = obj.numOutput;
        end
        
        function gradient = EvalBackpropNumerically(obj, dout)
            % Fully connected layers has 3 inputs so we have 3 gradients
            fcProp_x = @(x) obj.ForwardPropagation(x,obj.weights, obj.biases);
            fcProp_w = @(x) obj.ForwardPropagation(obj.previousInput,x, obj.biases);
            fcProp_b = @(x) obj.ForwardPropagation(obj.previousInput,obj.weights, x);
            
            % Evaluate
            gradient.input = GradientCheck.Eval(fcProp_x,obj.previousInput,dout);
            gradient.weight = GradientCheck.Eval(fcProp_w,obj.weights,dout);
            gradient.bias = GradientCheck.Eval(fcProp_b,obj.biases, dout);            
        end
    end
    
end

