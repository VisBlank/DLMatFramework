classdef MaxPoolLayer < BaseLayer
    %MaxPoolLayer Summary of this class goes here
    % Reference:
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    % https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolution_layer.html
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment3/cs231n/fast_layers.py#L13
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment3/cs231n/fast_layers.py#L106
    % http://sunshineatnoon.github.io/Using.Computation.Graph.to.Understand.and.Implement.Backpropagation/
    % https://uk.mathworks.com/matlabcentral/newsreader/view_thread/279051
    
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
        
        % Some data used for maxpool
        m_kernelHeight
        m_kernelWidth
        m_stride        
    end
    
    methods (Access = 'public')
        function [obj] = MaxPoolLayer(name, kH, kW, stride, index, inLayer)
            obj.name = name;
            obj.index = index;
            %obj.numOutput = numOutput;
            obj.inputLayer = inLayer;
            %obj.activationShape = [1 numOutput];
            obj.m_kernelHeight = kH;
            obj.m_kernelWidth = kW;
            obj.m_stride = stride;            
        end
        
        function [activations] = ForwardPropagation(obj, input, weights, bias)
            % Tensor format (rows,cols,channels, batch) on matlab
            [H,W,C,N] = size(input);
                        
            % Calculate output sizes
            H_prime = (H-obj.m_kernelHeight)/obj.m_stride +1;
            W_prime = (W-obj.m_kernelWidth)/obj.m_stride +1;
            
            % Alocate memory for output
            activations = zeros([H_prime,W_prime,C,N]);            
            %reshape so our im2col produces an output we can use
            im_split = reshape(input, H, W, 1, C*N); 
            im_col = im2col_ref_batch(im_split,obj.m_kernelHeight,obj.m_kernelWidth,obj.m_stride,0,0);
            
            %max pooling on each column (patch)   
            max_pool = max(im_col,[],1);
            
            %reshape to desired image output
            activations = reshape_row_major(max_pool,[H_prime W_prime C N]); 
                                    
            
            % Cache results for backpropagation
            obj.activations = activations;
            obj.weights = [];
            obj.biases = [];
            obj.previousInput = input;            
        end
        
        function [gradient] = BackwardPropagation(obj, dout)
            dout = dout.input;  
            [H,W,C,N] = size(obj.previousInput);            
            
            % Calculate output sizes
            H_prime = (H-obj.m_kernelHeight)/obj.m_stride +1;
            W_prime = (W-obj.m_kernelWidth)/obj.m_stride +1;                        
            
            % Initialize gradients            
            dx = zeros(size(obj.previousInput));            
            
            % Basically we just need to multiply dout by a mask created
            % from the cells that we selected on the previous forward
            % propagation

            %% Output gradients    
            gradient.bias = [];
            gradient.input = dx;  
            gradient.weight = [];
            
            if obj.doGradientCheck
                evalGrad = obj.EvalBackpropNumerically(dout);
                diff = sum(abs(evalGrad.input(:) - gradient.input(:)));                                               
                if diff > 1e-5
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

