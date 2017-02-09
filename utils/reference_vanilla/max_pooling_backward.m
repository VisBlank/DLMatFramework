%% Vanila Max pooling with n-dimensions
% Parameters:
% F: Kernel size
% S: stride

function dx = max_pooling_backward(dout, activations, Ky,Kx,S)
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
                x_pool = activations(initRow:endRow,initCol:endCol,depth,n);
                
                % Mask in only the biggest value on this window
                mask = x_pool == max(x_pool(:));
                
                % Apply this mask on dout, then accumulate
                prodMask = mask * dout(r,c,depth,n);
                dx(initRow:endRow,initCol:endCol,depth,n) = dx(initRow:endRow,initCol:endCol,depth,n) + prodMask;
            end
        end
    end
end
end

