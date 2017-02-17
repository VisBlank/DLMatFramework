classdef SpatialBatchNorm < BaseLayer
    %SpatialBatchNorm Does the spatial batchnorm (Between CONV and
    %activation function (Relu)).
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
        isTraining
        eps
        momentum
        running_mean = []
        running_var = []
        xhat
        xmu
        ivar
        sqrtvar
        var
        normalBatchNorm
    end
    
    methods (Access = 'public')
        function [obj] = SpatialBatchNorm(name, eps, momentum, index, inLayer)
            obj.name = name;
            obj.index = index;            
            obj.inputLayer = inLayer;
            
            % Create a batchnorm object
            obj.normalBatchNorm = BatchNorm(name, eps, momentum, index, inLayer);
            
            % Relu does not change the shape of it's output
            if ~isempty(inLayer)
                obj.activationShape = obj.inputLayer.getActivationShape();
            end
        end
        
        % It's just a call to the normal batchnorm but with some
        % permute/reshape on the input signal
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            [H,W,C,N] = size(input);
            
            % Permute the dimensions to the following format 
            % (cols, channel, rows, batch)    
            % On python was: x.transpose((0,2,3,1))
            % Python tensor format:
            % (batch(0), channel(1), rows(2), cols(3))
            % Matlab tensor format:
            % (rows(1), cols(2), channel(3), batch(4))
            inputTransposed = permute(input,[2,3,1,4]);                                    
            
            % Flat the input (On python the reshape is row-major)           
            inputFlat = reshape_row_major_custom(inputTransposed,[(numel(inputTransposed) / C),C]);
            
            % Call the forward propagation of normal batchnorm
            activations = obj.normalBatchNorm.ForwardPropagation(inputFlat, weights, bias);
            
            % Reshape/transpose back the signal, on python was (N,H,W,C)
            activations_reshape = reshape_row_major_custom(activations, [W,C,H,N]);
            % On python was transpose(0,3,1,2)
            activations = permute(activations_reshape,[3 1 2 4]);
            
            % Store stuff for backpropagation
            obj.activations = activations;
            obj.weights = weights;
            obj.biases = bias;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            % Observe that we use the same reshape/permutes from forward
            % propagation
            dout = dout.input;
            [H,W,C,N] = size(dout);
            % On python was: x.transpose((0,2,3,1))
            dout_transp = permute(dout,[2,3,1,4]);
            
            % Flat the input            
            dout_flat = reshape_row_major_custom(dout_transp,[(numel(dout_transp) / C),C]);
            
            % Call the backward propagation of normal batchnorm
            gradDout.input = dout_flat;
            gradient = obj.normalBatchNorm.BackwardPropagation(gradDout);
            
            % Reshape/transpose back the signal, on python was (N,H,W,C)
            gradient.input = reshape_row_major_custom(gradient.input, [W,C,H,N]);
            % On python was transpose(0,3,1,2)
            gradient.input = permute(gradient.input,[3 1 2 4]);
            
            % Evalulate numerically if needed
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
        
        function gradient = EvalBackpropNumerically(obj, dout)
            % Fully connected layers has 3 inputs so we have 3 gradients
            spatial_bn_x = @(x) obj.ForwardPropagation(x,obj.weights, obj.biases);  
            spatial_bn_gamma = @(x) obj.ForwardPropagation(obj.previousInput,x, obj.biases);
            spatial_bn_beta = @(x) obj.ForwardPropagation(obj.previousInput,obj.weights, x);                                    
            
            % Evaluate            
            gradient.input = GradientCheck.Eval(spatial_bn_x,obj.previousInput,dout);
            gradient.weight = GradientCheck.Eval(spatial_bn_gamma,obj.weights,dout);
            gradient.bias = GradientCheck.Eval(spatial_bn_beta,obj.biases, dout);
        end
        
        function IsTraining(obj, flag)
            obj.normalBatchNorm.IsTraining(flag);
            obj.isTraining = flag;
        end
    end
    
end

