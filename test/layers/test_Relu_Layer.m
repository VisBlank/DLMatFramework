clc; clear all;
load relu_forward_cs231n;

%% Test forward propagation
% Create Relu layer
rl = Relu('RL_1',[],[]);

activations = rl.ForwardPropagation(x, [], []);

% Compare results
diff = sum(abs(activations(:) - correct_out(:)));
if diff > 1e-6
    error('Relu forward pass failed');
else
    fprintf('Relu forward pass passed\n');    
end

%% Test backward propagation
clear all;
load relu_backward_cs231n;
gradDout.input = dout;

% Create Relu layer
rl = Relu('RL_1',[],[]);

% Do forward pass
activations = rl.ForwardPropagation(x, [], []);
% Do backward pass
gradients = rl.BackwardPropagation(gradDout);

% Compare differences
diff = sum(abs(gradients.input(:) - dx_num(:)));
if diff > 1e-8
    error('Relu backward pass failed');
else
    fprintf('Relu backward pass passed\n');    
end

