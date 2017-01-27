clc; clear all;
rng(0,'v5uniform');

%% Test forward propagation
x = randn(500,500) + 10;
prob = [0.3 0.6 0.75];

for idxProb=1:numel(prob)
    currProb = prob(idxProb);
    % Create instance of dropout layer
    dp = Dropout('DP_1',currProb,[],[]);
    dp.IsTraining(true);
    out = dp.ForwardPropagation(x,[],[]);
    dp.IsTraining(false);
    out_test = dp.ForwardPropagation(x,[],[]);        
    
    fprintf('Running tests with p = %d\n', currProb);
    fprintf('means of input: '); disp(mean(x(:)));
    fprintf('Mean of train-time output: '); disp(mean(out(:)));
    fprintf('Mean of test-time output: '); disp(mean(out_test(:)));
    fprintf('Fraction of train-time output set to zero: '); disp(mean2(out == 0));
    fprintf('Fraction of test-time output set to zero: '); disp(mean2(out_test == 0));
end

%% Test backward propagation
clear all;
load dropout_backward_cs231n
gradDout.input = dout;

dp = Dropout('DP_1',0.8,[],[]);
dp.IsTraining(true);

% Make python and matlab generate the same random numbers
% The only issue is that the results are transposed for matrices, so you
% need to manually transpose on the Dropout forward prop rand call (Disable
% this after the test).
rand('twister', 123);

out_mat = dp.ForwardPropagation(x,[],[]);
diff_db = sum(abs(out(:) - out_mat(:)));

% Will automatically raise an error if the gradient does not match with the
% numerical estimation
%dp.EnableGradientCheck(true);
gradients = dp.BackwardPropagation(gradDout);

% We cannot compare, if first we transpose the rand call on dropout forward
% propagation
% diff_dx = sum(abs(gradients.input(:) - dx_num(:)));
% 
% if diff_dx > 1e-8
%     error('Dropout backward pass failed');
% else
%     fprintf('Dropout forward pass passed\n');    
% end
