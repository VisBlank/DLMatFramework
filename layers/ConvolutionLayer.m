classdef ConvolutionLayer < BaseLayer
    %ConvolutionLayer Summary of this class goes here
    % Reference: 
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    % https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolution_layer.html
    
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
        
        % Some data used for convolution
        m_kernelHeight
        m_kernelWidth
        m_stride
        m_padding
    end
    
    methods (Access = 'public')
        function [obj] = ConvolutionLayer(name, kH, kW, stride, pad,index, inLayer)
            obj.name = name;
            obj.index = index;
            %obj.numOutput = numOutput;
            obj.inputLayer = inLayer;
            %obj.activationShape = [1 numOutput];
            obj.m_kernelHeight = kH;
            obj.m_kernelHeight = kW;
            obj.m_stride = stride;
            obj.m_padding = pad;
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)            
            
            % Tensor format (rows,cols,channels, batch) on matlab
            % Get batch size
            %[rows,cols,depth,N] = size(input);
            lenSizeActivations = length(size(input));
            if (lenSizeActivations < 3)
                N = size(input,1);
            else
                N = size(input,ndims(input));
            end
            
            
            
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
            
            
            
            if obj.doGradientCheck
                evalGrad = obj.EvalBackpropNumerically(dout);
                diff_Input = sum(abs(evalGrad.input(:) - gradient.input(:)));
                diff_Weights = sum(abs(evalGrad.weight(:) - gradient.weight(:)));
                diff_Bias = sum(abs(evalGrad.bias(:) - gradient.bias(:)));
                diff_vec = [diff_Input diff_Weights diff_Bias]; % diff_Bias
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
            convProp_x = @(x) obj.ForwardPropagation(x,obj.weights, obj.biases);
            convProp_w = @(x) obj.ForwardPropagation(obj.previousInput,x, obj.biases);
            convProp_b = @(x) obj.ForwardPropagation(obj.previousInput,obj.weights, x);
            
            % Evaluate
            gradient.input = GradientCheck.Eval(convProp_x,obj.previousInput,dout);
            gradient.weight = GradientCheck.Eval(convProp_w,obj.weights,dout);
            gradient.bias = GradientCheck.Eval(convProp_b,obj.biases, dout);            
        end
    end
    
end

