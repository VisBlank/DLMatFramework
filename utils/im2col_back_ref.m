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
    col = dout(:,ii);
    h_start = (ii / dout_W) * S;
    w_start = mod(ii,dout_W) * S;
    
    %patch = reshape_row_major(col,[HH,WW,C]);
    1+1;
end

end

