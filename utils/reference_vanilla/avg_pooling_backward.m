%% Vanila Max pooling with n-dimensions
% Parameters:
% F: Kernel size
% S: stride

function dx = avg_pooling_backward(dout, activations, Ky,Kx,S)
[H, W, C, N] = size(activations);
[HH,WW,~,~] = size(dout);
dx = zeros(size(activations));
% Calculate dx
for n=1:N
    for depth=1:C
        for r=1:HH
            for c=1:WW
                initRow = ((r-1)*S) + 1;
                endRow = ((r-1)*S+Ky);
                initCol = ((c-1)*S) + 1;
                endCol = ((c-1)*S+Kx);
    
                % We share dout equally among the input kernel for
                % averagepool
                dx(initRow:endRow,initCol:endCol,depth,n) = dx(initRow:endRow,initCol:endCol,depth,n) + (dout(r,c,depth,n)/(Ky*Kx));
            end
        end
    end
end
end

