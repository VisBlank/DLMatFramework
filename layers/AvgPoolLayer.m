classdef AvgPoolLayer < BaseLayer
    %AvgPoolLayer Summary of this class goes here
    % Reference:
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    % Average pooling backpropagation
    % https://youtu.be/cDdpwAIsuD8?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH&t=646
    
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
        
        % Some data used for avgpool
        m_kernelHeight
        m_kernelWidth
        m_stride
        selectedItems
        prevImcol
        m_canDoFast
        m_reshapedInputForFast
    end
    
    methods (Access = 'public')
        function [obj] = AvgPoolLayer(name, kH, kW, stride, index, inLayer)
            obj.name = name;
            obj.index = index;
            %obj.numOutput = numOutput;
            obj.inputLayer = inLayer;
            %obj.activationShape = [1 numOutput];
            obj.m_kernelHeight = kH;
            obj.m_kernelWidth = kW;
            obj.m_stride = stride;
            
            % Calculate the activation shape to be used to correctly
            % initialize the parameters of the next layers
            % Calculate output sizes
            if ~isempty(inLayer)
                inShape = inLayer.getActivationShape();
                H = inShape(1); W = inShape(2); C = inShape(3);
                H_prime = (H-kH)/stride +1;
                W_prime = (W-kW)/stride +1;
                obj.activationShape = [H_prime W_prime C -1];
            end
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
            
%             if same_kernel_size && tile
%                 % Can do the fast mode (vectorized max)
%                 obj.m_canDoFast = true;
%                 
%                 % Create a 6d tensor with the spatial dimensions divided,
%                 % for example if input is [4x4x3x2] the output of this
%                 % reshape will be [2x2x2x2x3x2]
%                 x_reshaped = reshape(input,[obj.m_kernelHeight, W/obj.m_kernelHeight, obj.m_kernelWidth,H/obj.m_kernelWidth,C,N]);
%                 
%                 % Get the biggest element along the row dimension of
%                 % x_reshaped then the biggest element along the third
%                 % dimension of this result, resulting on a 6d tensor
%                 avgpool_out = mean(mean(x_reshaped,[],1),[],3);
%                 
%                 % Reshape back again to the desired output activation shape
%                 activations = reshape(avgpool_out,[H_prime W_prime, C, N]);
%                 
%                 % Cache reshaped input
%                 obj.m_reshapedInputForFast = x_reshaped;
%             else
                % Fall back to slower (naive version), note that on caffe
                % this version is default (but written in C++)
                for n=1:N
                    input_batch = input(:,:,:,n);
                    resPool = avg_pooling_forward(input_batch,obj.m_kernelHeight, obj.m_kernelWidth,obj.m_stride);
                    activations(:,:,:,n) = resPool;
                end                
%             end
            
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
                dx_reshaped = zeros(size(obj.m_reshapedInputForFast));
                
                % Create a version of activation if added dimensions, so
                % for example if activations are [4x4x2x3] we would like
                % [4x1x4x1x2x3]
                activ_new_axis = reshape(obj.activations,[1 H_prime 1 W_prime C N]);
                
                % Get the mask
                % Do a repmat (lack of broadcast on matlab2016a) on
                % active_new_axis to match the dimensions of x_reshaped
                % Basically we want to repeat 2 the first and third
                % dimensions
                activ_new_axis = repmat(activ_new_axis,[2,1,2,1,1,1]);
                mask = (obj.m_reshapedInputForFast == activ_new_axis);
                
                dout_new_axis = reshape(dout,[1 H_prime 1 W_prime C N]);
                dout_new_axis = repmat(dout_new_axis,[2,1,2,1,1,1]);
                
                dx_reshaped(mask) = dout_new_axis(mask);
                
                % Reshape back to the input shape
                dx = reshape(dx_reshaped, size(obj.previousInput));
            else
                % Backpropagation in the case that we fall back to the
                % naive implementation
                dx = avg_pooling_backward(dout, obj.previousInput, obj.m_kernelHeight,obj.m_kernelWidth,obj.m_stride);                                
            end
            
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
            % Calculate numerical gradient
            convProp_x = @(x) obj.ForwardPropagation(x,obj.weights, obj.biases);            
            
            % Evaluate
            gradient.input = GradientCheck.Eval(convProp_x,obj.previousInput,dout);            
        end
    end
    
end

