clc; clear all;
load conv_forward_cs231n;

%% Permute tensors from python
% Input Tensor from python format (batch, channels, rows, cols)
% Input Tensor on matlab format (rows, cols, channels, batch)
x = permute(x,[3,4,2,1]);
out = permute(out,[3,4,2,1]);

% Weight Tensor from python format (num_filter, channels, k_rows, k_cols)
% Weight Tensor on matlab format (rows, cols, channels, num_filter)
w = permute(w,[3,4,2,1]);

% Transpose b
b = b';

% Get sizes
[k_rows,k_cols,C,F] = size(w);

% Create FC layer
conv = ConvolutionLayer('CONV_1',k_rows, k_cols, 2, 1,[],[]);

% Do forward pass
out_conv = conv.ForwardPropagation(x,w,b);

% Compare differences
diff = sum(abs(out_conv(:) - out(:)));
if diff > 1e-9
    error('CONV forward pass failed');
else
    fprintf('CONV forward pass passed\n');    
end

clear all;
load conv_backward_cs231n;
%% Permute tensors from python
% Input Tensor from python format (batch, channels, rows, cols)
% Input Tensor on matlab format (rows, cols, channels, batch)
x = permute(x,[3,4,2,1]);
w = permute(w,[3,4,2,1]);
b = b';
out = permute(out,[3,4,2,1]);

dout = permute(dout,[3,4,2,1]);
dx_num = permute(dx_num,[3,4,2,1]);
dw_num = permute(dw_num,[3,4,2,1]);
db_num = db_num';

% Put on the expected format for gradient
gradDout.input = dout;

% Get sizes
[k_rows,k_cols,C,F] = size(w);

% Create FC layer
conv = ConvolutionLayer('CONV_1',k_rows, k_cols, 1, 1,[],[]);

% Do forward pass
out_conv = conv.ForwardPropagation(x,w,b);
% Do backward pass
%conv.EnableGradientCheck(true);
gradients = conv.BackwardPropagation(gradDout);

% Compare differences
diff_db = sum(abs(gradients.bias(:) - db_num(:)));
diff_dw = sum(abs(gradients.weight(:) - dw_num(:)));
diff_dx = sum(abs(gradients.input(:) - dx_num(:)));
diff = sum([diff_db diff_dw diff_dx]);

if diff > 1e-5
    error('CONV backward pass failed');
else
    fprintf('CONV backward pass passed\n');    
end

