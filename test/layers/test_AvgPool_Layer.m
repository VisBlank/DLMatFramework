%% forward test

clc; clear all;

x = [1 5 7 8;8 8 8 8; 1 2 3 4; 9 5 1 1];
out = [5.5 7.75;4.25 2.25];
% Create AvgPool layer
avgpool = AvgPoolLayer('Avg_P1',2, 2, 2,[],[]);

% Do forward pass
poolOut = avgpool.ForwardPropagation(x,[],[]);

% Compare differences
diff = sum(abs(poolOut(:) - out(:)));
if diff > 1e-6
    error('AVGPOOL forward pass failed');
else
    fprintf('AVGPOOL forward pass passed\n');    
end



%% backward test

clc; clear all;

load maxpool_backward_cs231n;

x = permute(x,[3,4,2,1]);
dout = permute(dout,[3,4,2,1]);
dx_num = permute(dx_num,[3,4,2,1]);

% Put on the expected format for gradient
gradDout.input = dout;

% Create FC layer
avgpool = AvgPoolLayer('AVG_P1',2, 2, 2,[],[]);

% Do forward pass
poolOut = avgpool.ForwardPropagation(x,[],[]);

% Do backward pass
avgpool.EnableGradientCheck(true);
gradients = avgpool.BackwardPropagation(gradDout);

dx = gradients.input;

% Compare differences NEED CORRECT OUTPUT TO CHECK WITH
diff = sum(abs(gradients.input(:) - dx_num(:)));

% if diff > 1e-7
%     error('MAXPOOL backward pass failed');
% else
%     fprintf('MAXPOOL backward pass passed\n');    
% end