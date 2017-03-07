x1 = rand([100 50 25]);
x2 = rand([100 50 25]);

eal = EltWiseAdd('EW_1',[],[]);

activations = eal.ForwardPropagation(x1,x2);

actual = x1 + x2;

diff = sum(abs(activations(:) - actual(:)));
if diff > 1e-6
    error('EltWiseAdd forward pass failed');
else
    fprintf('EltWiseAdd forward pass passed\n');    
end