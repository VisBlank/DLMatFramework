classdef BatchNorm < BaseLayer
    %BatchNorm Summary of this class goes here
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
        running_mean
        running_var
        xhat
        xmu
        ivar
        sqrtvar
        var
    end
    
    methods (Access = 'public')
        function [obj] = BatchNorm(name, eps, momentum, index, inLayer)
            obj.name = name;
            obj.index = index;
            obj.eps = eps;
            obj.momentum = momentum;
            obj.isTraining = true;            
            obj.inputLayer = inLayer;
            % Relu does not change the shape of it's output
            obj.activationShape = obj.inputLayer.getActivationShape();
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            % Tensor format (rows,cols,channels, batch) on matlab
            % Get batch size            
            lenSizeActivations = length(size(input));
            if (lenSizeActivations < 3)
                N = size(input,1);
            else
                N = size(input,ndims(input));
            end
            
            if (obj.isTraining)
                shapeInput = size(input);
                
                % Step1: Calculate mean on the batch
                mu = (1/N) * sum(input,1);
                
                % Step2: Calculate mean on the batch
                obj.xmu = input - mu;
                
                % Step3: Calculate denominator
                sq = obj.xmu .^ 2;
                
                % Step4: Calculate variance
                obj.var = (1/N) * sum(sq,1);
                
                % Step5: add eps for numerical stability, then sqrt
                obj.sqrtvar = sqrt(obj.var + obj.eps);
                
                % Step6: Invert the square root
                obj.ivar = 1./obj.sqrtvar;
                
                %Step7: Do normalization
                obj.xhat = obj.xmu .* obj.ivar;
                
                %Step8: Nor the two transformation steps
                gammax = weights .* obj.xhat;
                
                % Step9: Adjust with bias (Batchnorm output)
                activations = gammax + bias;                                
            else
                
            end
            
            % Store stuff for backpropagation
            obj.activations = activations;
            obj.weights = weights;
            obj.biases = bias;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;
                        
            
            gradient.input = dx;
            
            if obj.doGradientCheck
                evalGrad = obj.EvalBackpropNumerically(dout);
                diff_Input = sum(abs(evalGrad.input(:) - gradient.input(:)));                
                diff_vec = [diff_Input]; 
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
            dropout_x = @(x) obj.ForwardPropagation(x,obj.weights, obj.biases);            
            
            % Evaluate
            gradient.input = GradientCheck.Eval(dropout_x,obj.previousInput, dout);            
        end
        
        function IsTraining(obj, flag)
            obj.isTraining = flag;
        end
    end
    
end

