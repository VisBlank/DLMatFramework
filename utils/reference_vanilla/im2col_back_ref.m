function [ img_grad ] = im2col_back_ref( dout, dout_H, dout_W, S, HH, WW, C )
%IM2COL_BACK_REF Backpropagation of im2col
% dout: (
% Return
% Image gradient (H,W,C)

% Calculate the spatial dimensions of im_grad
% Observe that the result will be "padded"
H = (dout_H - 1) * S + HH;
W = (dout_W - 1) * S + WW;

img_grad = zeros(H,W,C);

for ii=1:(dout_H*dout_W)
    row = dout(ii,:);
    
    % Create a patch from the row
    patch = reshape_row_major(row,[HH WW C]);
    %patch = reshape(row,[HH WW C]);
    
    % Calculate indexes on dx
    h_start = floor(((ii-1) / dout_W) * S);    
    w_start = mod((ii-1),dout_W) * S;
    h_start = h_start + 1;
    w_start = w_start + 1;
        
    img_grad(h_start:h_start+HH-1, w_start:w_start+WW-1, :) = img_grad(h_start:h_start+HH-1, w_start:w_start+WW-1, :) + patch;    
end

end

