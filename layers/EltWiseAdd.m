classdef EltWiseAdd < BaseLayer
    %EltWiseAdd Summary of this class goes here
    
    
    properties (Access = 'protected') 
        weights
        biases
        activations
        config
        previousInput
        previousInput1
        previousInput2
        name
        index
        activationShape
        inputLayer
        additionResult
    end
    
    methods (Access = 'public')
        function [obj] = EltWiseAdd(name, index, inLayer)
            obj.name = name;
            obj.index = index;
            obj.inputLayer = inLayer;
            % EltWiseAdd does not change the shape of it's output
            if ~isempty(inLayer)
                obj.activationShape = obj.inputLayer.getActivationShape();
            end
        end
        
        function [additionResult] = ForwardPropagation(obj, input1, input2)
            
            obj.previousInput1 = input1;
            obj.previousInput2 = input2;
            additionResult = input1 + input2;
            obj.additionResult = additionResult;
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;
            dx = dout ;
            gradient.input1 = dx;
            gradient.input2 = dx;
            
            if obj.doGradientCheck
                evalGrad = obj.EvalBackpropNumerically(dout);
                diff_Input1 = sum(abs(evalGrad.input1(:) - gradient.input1(:)));   
                diff_Input2 = sum(abs(evalGrad.input2(:) - gradient.input2(:)));  
                diff_vec = [diff_Input1 diff_Input2]; 
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
            % Eltwise connected layers has 2 inputs so we have 2 gradients
            eltwiseadd_x1 = @(x) obj.ForwardPropagation(x, obj.previousInput1, obj.previousInput2 );            
            eltwiseadd_x2 = @(x) obj.ForwardPropagation(x, obj.previousInput1, obj.previousInput2 );
            
            % Evaluate
            gradient.input1 = GradientCheck.Eval(eltwiseadd_x1,obj.previousInput1, dout);
            gradient.input2 = GradientCheck.Eval(eltwiseadd_x2,obj.previousInput2, dout);
        end
    end
    
end

