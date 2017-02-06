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
        running_mean = []
        running_var = []
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
            if ~isempty(inLayer)
                obj.activationShape = obj.inputLayer.getActivationShape();
            end
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            % Tensor format (rows,cols,channels, batch) on matlab
            % Get batch size            
            lenSizeActivations = length(size(input));
            [~,D] = size(input);
            if (lenSizeActivations < 3)
                N = size(input,1);
            else
                N = size(input,ndims(input));
            end
            
            % Initialize for the first time running_mean and running_var
            if isempty(obj.running_mean)
                obj.running_mean = zeros(1,D);
                obj.running_var = zeros(1,D);
            end
            
            if (obj.isTraining)                                
                % Step1: Calculate mean on the batch
                mu = (1/N) * sum(input,1);
                
                % Step2: Subtract the mean from each column
                obj.xmu = input - repmat(mu,N,1);
                
                % Step3: Calculate denominator
                sq = obj.xmu .^ 2;
                
                % Step4: Calculate variance
                obj.var = (1/N) * sum(sq,1);
                
                % Step5: add eps for numerical stability, then sqrt
                obj.sqrtvar = sqrt(obj.var + obj.eps);
                
                % Step6: Invert the square root
                obj.ivar = 1./obj.sqrtvar;
                
                %Step7: Do normalization
                obj.xhat = obj.xmu .* repmat(obj.ivar,N,1);
                
                %Step8: Nor the two transformation steps
                gammax = repmat(weights,N,1) .* obj.xhat;
                
                % Step9: Adjust with bias (Batchnorm output)
                activations = gammax + repmat(bias,N,1); 
                
                % Calculate running mean and variance to be used latter on
                % prediction
                obj.running_mean = (obj.momentum .* obj.running_mean) + (1.0 - obj.momentum) * mu;
                obj.running_var = (obj.momentum .* obj.running_var) + (1.0 - obj.momentum) .* obj.var;
            else
                xbar = (input - repmat(obj.running_mean,N,1)) ./ repmat(sqrt(obj.running_var + obj.eps),N,1);
                activations = (repmat(weights,N,1) .* xbar) + repmat(bias,N,1);
            end
            
            % Store stuff for backpropagation
            obj.activations = activations;
            obj.weights = weights;
            obj.biases = bias;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;
            lenSizeActivations = length(size(obj.previousInput));
            [~,D] = size(obj.previousInput);
            if (lenSizeActivations < 3)
                N = size(obj.previousInput,1);
            else
                N = size(obj.previousInput,ndims(obj.previousInput));
            end
            
            % Step9:
            dbeta = sum(dout, 1);
            dgammax = dout;
            
            % Step8:
            dgamma = sum(dgammax.*obj.xhat, 1);
            dxhat = dgammax .* repmat(obj.weights,N,1);
            
            % Step7:
            divar = sum(dxhat.* obj.xmu, 1);
            dxmu1 = dxhat .* repmat(obj.ivar,N,1);
            
            % Step6:
            dsqrtvar = -1 ./ (obj.sqrtvar.^2) .* divar;
                        
            % Step 5:
            dvar = 0.5 * 1 ./sqrt(obj.var+obj.eps) .* dsqrtvar;
            
            % Step 4:
            dsq = 1 ./ N * ones(N,D) .* repmat(dvar,N,1);
            
            % Step 3:
            dxmu2 = 2 .* obj.xmu .* dsq;
            
            % Step 2:
            dx1 = (dxmu1 + dxmu2);
            dmu = -1 .* sum(dxmu1+dxmu2, 1);
            
            % Step 1:
            dx2 = 1. /N .* ones(N,D) .* repmat(dmu,N,1);
            
            gradient.input = dx1+dx2;
            gradient.weight = dgamma;
            gradient.bias = dbeta;
            
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
            bn_x = @(x) obj.ForwardPropagation(x,obj.weights, obj.biases);  
            bn_gamma = @(x) obj.ForwardPropagation(obj.previousInput,x, obj.biases);
            bn_beta = @(x) obj.ForwardPropagation(obj.previousInput,obj.weights, x);                                    
            
            % Evaluate            
            gradient.input = GradientCheck.Eval(bn_x,obj.previousInput,dout);
            gradient.weight = GradientCheck.Eval(bn_gamma,obj.weights,dout);
            gradient.bias = GradientCheck.Eval(bn_beta,obj.biases, dout);
        end
        
        function IsTraining(obj, flag)
            obj.isTraining = flag;
        end
    end
    
end

