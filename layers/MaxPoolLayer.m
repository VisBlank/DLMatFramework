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
        selectedItems
        prevImcol
        m_canDoFast
        m_reshapedInputForFast
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
            
            %% Decide between im2col or fast implementation
            same_kernel_size = (obj.m_kernelHeight == obj.m_kernelWidth);
            tile = (mod(H,obj.m_kernelHeight) == 0) && (mod(H,obj.m_kernelWidth) == 0);
            if same_kernel_size && tile
                % Can do the fast mode (vectorized max)
                obj.m_canDoFast = true;
                
                % Create a 6d tensor with the spatial dimensions divided,
                % for example if input is [4x4x3x2] the output of this
                % reshape will be [2x2x2x2x3x2]
                x_reshaped = reshape(input,[obj.m_kernelHeight, W/obj.m_kernelHeight, obj.m_kernelWidth,H/obj.m_kernelWidth,C,N]);
                
                % Get the biggest element along the row dimension of
                % x_reshaped then the biggest element along the third
                % dimension of this result, resulting on a 6d tensor                
                maxpool_out = max(max(x_reshaped,[],1),[],3);
                
                % Reshape back again to the desired output activation shape
                activations = reshape(maxpool_out,[H_prime W_prime, C, N]);
                
                % Cache reshaped input
                obj.m_reshapedInputForFast = x_reshaped;
            else
                % Fall back to im2col or naive
                %reshape so our im2col produces an output we can use
                im_split = reshape(input, H, W, 1, C*N);
                im_col = im2col_ref_batch(im_split,obj.m_kernelHeight,obj.m_kernelWidth,obj.m_stride,0,0);
                
                %max pooling on each column (patch)
                [max_pool, idxSelected] = max(im_col,[],1);
                
                %reshape to desired image output
                activations = reshape_row_major(max_pool,[H_prime W_prime C N]);
                
                obj.selectedItems = idxSelected;
                obj.prevImcol = im_col;
            end
            
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
            
            %% The backpropagation will depend on mode used on forward prop
            if obj.m_canDoFast
                
            end
            
            % Reshape dout (
            dout_reshaped = permute(dout,[2,1,4,3]);
            dout_reshaped = dout_reshaped(:);
            dx_cols = zeros(size(obj.prevImcol));
            %dx_cols(obj.selectedItems, :) = dout_reshaped;
            
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

