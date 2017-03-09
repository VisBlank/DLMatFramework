classdef Dropout < BaseLayer
    %Dropout Summary of this class goes here
    % Reference: https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    % http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DropoutLayer.html#af3d3f94306230950edf514e0fbb8f710
    % https://github.com/pfnet/chainer/blob/a6a4e373071c6be3215bdc1367cb3d40fbcd8a2a/chainer/functions/noise/dropout.py
    
    properties (Access = 'protected') 
        weights
        biases
        activations
        gradients
        config
        previousInput
        name
        index
        activationShape
        weightShape
        inputLayer
        dropoutMask
        dropoutProb
        isTraining
    end
    
    methods (Access = 'public')
        function [obj] = Dropout(name, prob, index, inLayer)
            obj.name = name;
            obj.index = index;
            obj.isTraining = true;
            obj.dropoutProb = prob;
            obj.inputLayer = inLayer;
            if ~isempty(inLayer)
                obj.activationShape = obj.inputLayer.getActivationShape();
            end
            obj.weightShape = [];
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            obj.previousInput = input;
            
            if (obj.isTraining)
                shapeInput = size(input);
                
                % If we're using gradient check we need to use a fixed
                % seed, but this is actually not needed if you are not
                % using gradient check
                if obj.doGradientCheck
                    rand('twister', 123);
                end
                
                % Inverted dropout (Forward during prediction will be
                % transparent)
                %obj.dropoutMask = (rand(shapeInput) < obj.dropoutProb) / obj.dropoutProb;                
                obj.dropoutMask = (rand(shapeInput) >= obj.dropoutProb) ./ (1-obj.dropoutProb);
                % Uncomment if you want to compare with python results
                % (rand from python has results transposed w.r.t to matlab)
                %obj.dropoutMask = (rand(shapeInput)' >= obj.dropoutProb) ./ (1-obj.dropoutProb);
                
                activations = input .* obj.dropoutMask;
            else
                activations = input;
                obj.dropoutMask = [];
            end
            
            obj.activations = activations;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;
            
            % During backprop use the same mask to mask from training on
            % the input gradient
            dx = dout .* obj.dropoutMask;
            
            gradient.input = dx;
            
            % Cache gradients
            obj.gradients = gradient;
            
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

