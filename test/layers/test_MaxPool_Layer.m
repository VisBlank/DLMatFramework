clc; clear all;
load maxpool_forward_cs231n;

%% Permute tensors from python
% Input Tensor from python format (batch, channels, rows, cols)
% Input Tensor on matlab format (rows, cols, channels, batch)
x = permute(x,[3,4,2,1]);
out = permute(out,[3,4,2,1]);
correct_out = permute(correct_out,[3,4,2,1]);

% Create FC layer
maxpool = MaxPoolLayer('MAX_P1',2, 2, 2,[],[]);

% Do forward pass
poolOut = maxpool.ForwardPropagation(x,[],[]);

% Compare differences
diff = sum(abs(poolOut(:) - out(:)));
if diff > 1e-6
    error('MAXPOOL forward pass failed');
else
    fprintf('MAXPOOL forward pass passed\n');    
end

clear all;
load maxpool_backward_cs231n;

%% Permute tensors from python
% Input Tensor from python format (batch, channels, rows, cols)
% Input Tensor on matlab format (rows, cols, channels, batch)
x = permute(x,[3,4,2,1]);
dout = permute(dout,[3,4,2,1]);
dx_num = permute(dx_num,[3,4,2,1]);

% Put on the expected format for gradient
gradDout.input = dout;

% Create FC layer
maxpool = MaxPoolLayer('MAX_P1',2, 2, 2,[],[]);

% Do forward pass
poolOut = maxpool.ForwardPropagation(x,[],[]);

% Do backward pass
maxpool.EnableGradientCheck(true);
gradients = maxpool.BackwardPropagation(gradDout);

% Compare differences
diff = sum(abs(gradients.input(:) - dx_num(:)));

if diff > 1e-7
    error('MAXPOOL backward pass failed');
else
    fprintf('MAXPOOL backward pass passed\n');    
end