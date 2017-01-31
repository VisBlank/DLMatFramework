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

