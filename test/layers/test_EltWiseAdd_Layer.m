x1 = rand([10 5 2]);
x2 = rand([10 5 2]);
dout_vals = x1 * 0.01;

eal = EltWiseAdd('EW_1',[],[]);

activations = eal.ForwardPropagation(x1,x2);

actual = x1 + x2;

diff = sum(abs(activations(:) - actual(:)));
if diff > 1e-6
    error('EltWiseAdd forward pass failed');
else
    fprintf('EltWiseAdd forward pass passed\n');    
end

%% Check backprop (gradient check only for now)
dout.input = dout_vals;
eal.EnableGradientCheck(true);
eal.BackwardPropagation(dout);