clc; clear all;
load fc_forward_cs231n;

%% Permute tensors from python
% Tensor from python format (batch, channels, rows, cols)
% Tensor on matlab format (rows, cols, channels, batch)
x = permute(x,[3,4,2,1]);
numOutput = size(w,2);

% Create FC layer
fc = FullyConnected('FC_1',numOutput,[],[]);

% Do forward pass
out = fc.ForwardPropagation(x,w,b);

% Compare differences
diff = sum(abs(out(:) - correct_out(:)));
if diff > 1e-6
    error('FC forward pass failed');
else
    fprintf('FC forward pass passed\n');
    disp(out);
end

clear all;
load fc_backward_cs231n;
gradDout.input = dout;
numOutput = size(w,2);
x = permute(x,[3,4,2,1]);
dx_num = permute(dx_num,[3,4,2,1]);

% Create FC layer
fc = FullyConnected('FC_1',numOutput,[],[]);

% Do forward pass
out = fc.ForwardPropagation(x,w,b);

% Do backward pass
fc.EnableGradientCheck(true);
gradients = fc.BackwardPropagation(gradDout);

% Compare differences
diff_db = sum(abs(gradients.bias(:) - db_num(:)));
diff_dw = sum(abs(gradients.weight(:) - dw_num(:)));
diff_dx = sum(abs(gradients.input(:) - dx_num(:)));
diff = sum([diff_db diff_dw diff_dx]);
if diff > 1e-8
    error('FC backward pass failed');
else
    fprintf('FC forward pass passed\n');
    disp(out);
end
