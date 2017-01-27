clc; clear all;
load batchnorm_forward_cs231n;

%% Test forward propagation
% Create Batchnorm layer
bn = BatchNorm('BN_1',1e-5,0.9,[],[]);

disp('Before batch normalization:');
fprintf('a.shape(): '); disp(size(a));
fprintf('means: '); disp(mean(a,1));
fprintf('std: '); disp(std(a,1));

activations = bn.ForwardPropagation(a, ones(1,3), zeros(1,3));

fprintf('\n\nAfter batch normalization (gamma=1, beta=0) Means should be close to zero and std close to one\n');
fprintf('means: '); disp(round(mean(activations,1)));
fprintf('std: '); disp(std(activations,1));

% Compare results
diff = sum(abs(activations(:) - a_norm_resp_1(:)));
if diff > 1e-6
    error('BN forward pass failed');
else
    fprintf('BN forward pass passed\n');    
end

%% Test backward propagation
clear all;
load batchnorm_backward_cs231n;
gradDout.input = dout;

disp('Before batch normalization:');
fprintf('x.shape(): '); disp(size(x));
fprintf('means: '); disp(mean(x,1));
fprintf('std: '); disp(std(x,1));

% Create Batchnorm layer
bn = BatchNorm('BN_1',1e-5,0.9,[],[]);

% Do forward pass
bn.ForwardPropagation(x, gamma, beta);
% Do backward pass
gradients = bn.BackwardPropagation(gradDout);

% Compare differences
diff_db = sum(abs(gradients.bias(:) - d_beta_num(:)));
diff_dw = sum(abs(gradients.weight(:) - d_gama_num(:)));
diff_dx = sum(abs(gradients.input(:) - d_x_num(:)));
diff = sum([diff_db diff_dw diff_dx]);
if diff > 1e-8
    error('BN backward pass failed');
else
    fprintf('BN backward pass passed\n');    
end

