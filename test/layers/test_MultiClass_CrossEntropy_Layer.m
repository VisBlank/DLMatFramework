clc; clear all;
load multi_class_cross_entropy_cs231n;

%% Convert targets y to one-hot
y = y';
valueLabels = unique(y);
nLabels = length(valueLabels);
nSamples = size(y,1);
targets = zeros(nSamples, nLabels);
for i = 1:nLabels
    targets(:,i) = (y == valueLabels(i));
end

%% Test forward propagation
% Create Softmax layer and Multi-class cross entropy loss function
sm = Softmax('SM_1',[],[]);
lossFunc = LossFactory.GetLoss('multi_class_cross_entropy');

scores = sm.ForwardPropagation(x, [], []);
[data_loss, dx] = lossFunc.GetLossAndGradients(scores, targets);

% Compare results
diff_loss = sum(abs(data_loss(:) - loss(:)));
diff_dx = sum(abs(dx(:) - dx_num(:)));
diff = sum([diff_loss diff_dx]);
if diff > 1e-6
    error('Loss with Softmax backward pass failed');
else
    fprintf('Loss with Softmax backward pass passed\n');
end
