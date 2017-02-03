classdef MaxPoolLayer < BaseLayer
    %MaxPoolLayer Summary of this class goes here
    % Reference:
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/layers.py
    % https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolution_layer.html
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment3/cs231n/fast_layers.py#L13
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment3/cs231n/fast_layers.py#L106
    % http://sunshineatnoon.github.io/Using.Computation.Graph.to.Understand.and.Implement.Backpropagation/
    % https://uk.mathworks.com/matlabcentral/newsreader/view_thread/279051
    % Mastering matrix indexing in matlab
    % https://uk.mathworks.com/company/newsletters/articles/matrix-indexing-in-matlab.html
    
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
            tile = 0;
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
%                 im_split = reshape(input, H, W, 1, C*N);
%                 im_col = im2col_ref_batch(im_split,obj.m_kernelHeight,obj.m_kernelWidth,obj.m_stride,0,0);
%                 
%                 %max pooling on each column (patch)
%                 [max_pool, idxSelected] = max(im_col,[],1);
%                 
%                 %reshape to desired image output
%                 activations = reshape_row_major(max_pool,[H_prime W_prime C N]);
%                 
%                 obj.selectedItems = idxSelected;
%                 obj.prevImcol = im_col;
                
                % Way of doing FP following the Convolution idea of using a
                % im2col that works on each channel (Not complete batch
                % im2col)
                kH = obj.m_kernelHeight;
                kW = obj.m_kernelWidth;
                idxSelected = zeros(N*C,H_prime*W_prime);
                rowCount = 1;
                for idxBatch = 1:N
                    im = input(:,:,:,idxBatch);
                    im_col = im2col_ref(im,kH,kW,obj.m_stride,0,0);
                    % Iterate on each channel
                    stRowChan = 1;
                    edRowChan = kH*kW;
                    for idxChan=1:C                        
                        [max_pool, idxSelectedChannel] = max(im_col(stRowChan:edRowChan,:),[],1);                        
                        stRowChan = edRowChan+1;
                        edRowChan = (kH*kW) * (idxChan+1);
                        activations(:,:,idxChan,idxBatch) =  reshape_row_major(max_pool,[H_prime W_prime size(max_pool,1)]);
                        idxSelected(rowCount,:) = idxSelectedChannel;
                        rowCount = rowCount + 1;
                    end                                        
                end
                obj.selectedItems = idxSelected;
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
                
                % Reshape back the the input shape
                dx = reshape(dx_reshaped, size(obj.previousInput));
            else
                % Backpropagation on the case that we fall back to the
                % im2col implementation
                %dout_reshape = permute(dout,[3,4,1,2]);
                %dout_reshape = dout_reshape(:);
                %dx_cols = zeros(size(obj.prevImcol));
                
                % Set on dx_cols the values of dout_shape at the positions
                % that the forward propagation found max values
                %dx_cols(sub2ind(size(dx_cols), obj.selectedItems, [1:size(dx_cols,2)])) = dout_reshape;
                
                % Now we need to convert the im2col back to the image
                % format (col2im)
                %dx = zeros(H,W,1,N*C);
                %dx = col2im_batch_ref(dx_cols,H,W,1,N*C,obj.m_kernelHeight, obj.m_kernelWidth,0,obj.m_stride);
                
                % Now we need to reshape back dx to the shape of the input
                %dx = reshape(dx,size(obj.previousInput));
                
                % Way of doing BP following the Convolution idea of using a
                % im2col that works on each channel (Not complete batch
                % im2col) 
                kH = obj.m_kernelHeight;
                kW = obj.m_kernelWidth;
                rowCount = 1;
                for idxBatch = 1:N                                            
                    stRowChan = 1;
                    edRowChan = kH*kW;
                    for idxChan=1:C
                        dx_im2col = zeros(kH*kW*1,H_prime*W_prime);   
                        % Reshape dout
                        dout_i = dout(:,:,idxChan,idxBatch);                                                
                        dout_i_reshaped = reshape_row_major(dout_i,[1, H_prime*W_prime]);
                        
                        % Apply mask and populate with values from dout
                        dx_im2col(sub2ind(size(dx_im2col), obj.selectedItems(rowCount,:), [1:size(dx_im2col,2)])) = dout_i_reshaped;
                        
                        % results will padded by one always
                        dx_padded = im2col_back_ref(dx_im2col,H_prime, W_prime, obj.m_stride, kH, kW, 1);                
                        
                        stRowChan = edRowChan+1;
                        edRowChan = (kH*kW) * (idxChan+1);
                        rowCount = rowCount + 1;
                    end
                end
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

