clc; clear all;
load spatial_batchnorm_forward;

%% Permute tensors from python
% Input Tensor from python format (batch, channels, rows, cols)
% Input Tensor on matlab format (rows, cols, channels, batch)
x = permute(x,[3,4,2,1]);
out = permute(out,[3,4,2,1]);

%% Test forward propagation
% Create Batchnorm layer
bn = SpatialBatchNorm('SP_BN_1',1e-5,0.9,[],[]);

disp('Before batch normalization:');
% Means from axis batch,rows,cols
mean_rows_cols_batch = mean(mean(mean(x, 1), 2), 4);
mean_rows_cols_batch = reshape(mean_rows_cols_batch,[1,3]);
std_rows_cols_batch = std(std(std(x,0,1),0,2),0,4);
std_rows_cols_batch = reshape(std_rows_cols_batch,[1,3]);
fprintf('x.shape(): '); disp(size(x));
fprintf('means: '); disp(mean_rows_cols_batch); 
%fprintf('std: '); disp(std(x,1));

activations = bn.ForwardPropagation(x, gamma, beta);

mean_rows_cols_batch_act = mean(mean(mean(activations, 1), 2), 4);
mean_rows_cols_batch_act = reshape(mean_rows_cols_batch_act,[1,3]);
fprintf('means: '); disp(round(mean_rows_cols_batch_act));
%fprintf('std: '); disp(std(activations,1));

% Compare results
diff = sum(abs(activations(:) - out(:)));
if diff > 1e-6
    error('BN forward pass failed');
else
    fprintf('BN forward pass passed\n');    
end

%% Test backward propagation
% clear all;
% load batchnorm_backward_cs231n;
% gradDout.input = dout;
% 
% disp('Before batch normalization:');
% fprintf('x.shape(): '); disp(size(x));
% fprintf('means: '); disp(mean(x,1));
% fprintf('std: '); disp(std(x,1));
% 
% % Create Batchnorm layer
% bn = BatchNorm('BN_1',1e-5,0.9,[],[]);
% 
% % Do forward pass
% bn.ForwardPropagation(x, gamma, beta);
% % Do backward pass
% bn.EnableGradientCheck(true);
% gradients = bn.BackwardPropagation(gradDout);
% 
% % Compare differences
% diff_db = sum(abs(gradients.bias(:) - d_beta_num(:)));
% diff_dw = sum(abs(gradients.weight(:) - d_gama_num(:)));
% diff_dx = sum(abs(gradients.input(:) - d_x_num(:)));
% diff = sum([diff_db diff_dw diff_dx]);
% if diff > 1e-8
%     error('BN backward pass failed');
% else
%     fprintf('BN backward pass passed\n');    
% end
% 
