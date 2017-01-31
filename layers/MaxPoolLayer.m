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
            [H,W,~,N] = size(obj.previousInput);
            [HH,WW,C,F] = size(obj.weights);  
            
            % Calculate output sizes
            H_prime = (H+2*obj.m_padding-HH)/obj.m_stride +1;
            W_prime = (W+2*obj.m_padding-WW)/obj.m_stride +1;
            
            % Preparing filter weights            
            filter_col_T = reshape_row_major(obj.weights,[F HH*WW*C]);                                                
            
            % Initialize gradients
            dw = zeros(size(obj.weights));
            dx = zeros(size(obj.previousInput));            
            % Get the bias gradient which will be the sum of dout over the
            % dimensions (batches(4), rows(1), cols(2))
            db = sum(sum(sum(dout, 1), 2), 4);
            
            for idxBatch = 1:N
                im = obj.previousInput(:,:,:,idxBatch);    
                im_col = im2col_ref(im,HH,WW,obj.m_stride,obj.m_padding,1);
                dout_i = dout(:,:,:,idxBatch);
                                                
                dout_i_reshaped = reshape_row_major(dout_i,[F, H*W]);                
                
                dw_before_reshape = dout_i_reshaped * im_col';                
                dw_i = reshape(dw_before_reshape',[HH, WW, C, F]);
                dw = dw + dw_i;
                
                % We now have the gradient just before the im2col
                grad_before_im2col = (dout_i_reshaped' * filter_col_T);                
                % Now we need to backpropagate im2col (im2col_back),
                % results will padded by one always
                dx_padded = im2col_back_ref(grad_before_im2col,H_prime, W_prime, obj.m_stride, HH, WW, C);                
                % Now we need to take out the pading
                dx(:,:,:,idxBatch) = dx_padded(2:end-1, 2:end-1,:);
            end

            %% Output gradients    
            gradient.bias = db;
            gradient.input = dx;  
            gradient.weight = dw;
            
            if obj.doGradientCheck
                evalGrad = obj.EvalBackpropNumerically(dout);
                diff_Input = sum(abs(evalGrad.input(:) - gradient.input(:)));
                diff_Weights = sum(abs(evalGrad.weight(:) - gradient.weight(:)));
                diff_Bias = sum(abs(evalGrad.bias(:) - gradient.bias(:)));
                diff_vec = [diff_Input diff_Weights diff_Bias]; % diff_Bias
                diff = sum(diff_vec);
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

